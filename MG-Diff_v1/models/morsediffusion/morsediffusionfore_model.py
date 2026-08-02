from models import init_net, BaseModel
import torch
import torch.nn.functional as F
from .stformer import STFormerForecasting
from utils.util import _mae_with_missing,_rmse_with_missing, _mape_with_missing, _quantile_CRPS_with_missing, crps_mc_loss
from .model_util import get_schedule, laplacian_positional_encoding, temporal_positional_embedding, norm_adj
from .gwavenet import GWaveNetEncoder
import os
import numpy as np
import time
from .morse_function import (
    canon_edge,
    build_sbm_with_degree_valid_dmf_greedy_pair,
    identify_critical_cells_from_attrs,
    make_attr_morse_function,
    critical_covering_walk_high_to_low,
    run_complete_pipeline_with_visuals,
    get_critical_path
)
import networkx as nx

def sample_random_control(
    adj,
    num_critical_edges,
    num_critical_nodes,
    num_walk_edges,
    seed,
):
    """Construct a random structural control matched to Morse statistics.

    All sampled edges come from the original graph.
    """
    rng = np.random.default_rng(seed)

    adj_np = adj.detach().cpu().numpy()
    adj_binary = np.logical_or(adj_np > 0, adj_np.T > 0)

    original_edges = [
        (u, v)
        for u in range(adj_binary.shape[0])
        for v in range(u + 1, adj_binary.shape[1])
        if adj_binary[u, v]
    ]

    if num_critical_edges > len(original_edges):
        raise ValueError(
            f"Requested {num_critical_edges} critical edges, "
            f"but the graph contains only {len(original_edges)} edges."
        )

    if num_walk_edges > len(original_edges):
        raise ValueError(
            f"Requested {num_walk_edges} walk edges, "
            f"but the graph contains only {len(original_edges)} edges."
        )

    crit_indices = rng.choice(
        len(original_edges),
        size=num_critical_edges,
        replace=False,
    )
    random_critical_edges = [
        original_edges[i] for i in crit_indices
    ]

    # Independently match the number of critical nodes.
    random_critical_nodes = rng.choice(
        adj_binary.shape[0],
        size=num_critical_nodes,
        replace=False,
    ).tolist()

    walk_indices = rng.choice(
        len(original_edges),
        size=num_walk_edges,
        replace=False,
    )
    random_walk_edges = [
        original_edges[i] for i in walk_indices
    ]

    return (
        random_critical_edges,
        random_critical_nodes,
        random_walk_edges,
    )

class morsediffusionForeModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--morse_control",
            type=str,
            default="morse",
            choices=["morse", "random"],
            help="Use the true Morse skeleton or a size-matched random graph control.",
        )
        parser.add_argument(
            "--morse_seed",
            type=int,
            default=42,
            help="Random seed used only to construct the structural control.",
        )
        return parser

    def __init__(self, opt, model_config):
        super().__init__(opt, model_config)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['crps']
        self.loss_names = ['l2']
        # self.loss_names = ['mae']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['STD', 'Encoder']

        # specify metrics you want to evaluate the model. The training/test scripts will call functions in order:
        # <BaseModel.compute_metrics> compute metrics for current batch
        # <BaseModel.get_current_metrics> compute and return mean of metrics, clear evaluation cache for next evaluation
        self.metric_names = ['MAE', 'RMSE', 'MAPE']
        if self.opt.phase == 'test':
            self.metric_names += ['CRPS']

        # define networks. The model variable name should begin with 'self.net'
        model_config['task'] = 'forecasting'
        model_config['condition_dim'] = model_config['wavenet']['end_dim']
        model_config['input_dim'] = opt.y_dim + opt.covariate_dim
        model_config['t_len'] = opt.t_len // 2
        model_config['output_dim'] = opt.t_len // 2
        model_config['num_nodes'] = opt.num_nodes
        model_config['wavenet']['input_dim'] = opt.y_dim + opt.covariate_dim

        self.netEncoder = GWaveNetEncoder(model_config['wavenet'])
        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        # self.netEncoder.eval()

        self.netSTD = STFormerForecasting(model_config)
        self.netSTD = init_net(self.netSTD, opt.init_type, opt.init_gain, opt.gpu_ids)

        # parameters for diffusion models
        self.num_steps = model_config["num_steps"]
        self.beta = get_schedule(self.num_steps, model_config["schedule"])
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.alphas_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.)
        self.num_sample = model_config["num_sample"]

        # other parameters
        self.pos_dim = model_config['pos_dim']
        self.objective = model_config['objective'] # recover the original data or sampled noise

        # define loss functions
        if self.isTrain:
            self.criterion = self.l2_loss
            # self.criterion = self.crps_loss
            # self.criterion = self.mae_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.AdamW([{'params': self.netSTD.parameters()},
                                                {'params': self.netEncoder.parameters(), 'lr': opt.lr * 1}]
                                                ,lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)
            self.load_networks()  # only load pre-trained model
        
        # self.edge_crit_gate = torch.nn.Parameter(torch.tensor(-2.0))

    def set_input(self, input):
        """
        parse input for one epoch. data should be stored as self.xxx which would be adopted in self.forward().
        we construct the spatial embedding vectors here.
        :param input: dict
        :return: None
        """
        
        # network inputs
        self.pred_gt = input['pred'].to(self.device)  # [B, N, L, D]
        self.missing_mask = input['missing_mask'].to(self.device)
        if 'feat' in input.keys():
            self.covariate = input['feat'].to(self.device)  # [batch, num_n, time, d_c]

        # Critical points
        batch_size = input['adj'].shape[0]
        num_nodes = input['adj'].shape[1]

        #########
        #the following parts actually only run once in the training process
        #########
        if not hasattr(self, 'pos_enc'):

            adj = input['adj'][0]
            G_adj = (adj > 0).to(torch.float32).cpu().numpy()
            G, _, _ = build_sbm_with_degree_valid_dmf_greedy_pair(
                G_adj,
                seed=0,
                pair_rate=0.9,
                chord_gap=10.0,
            )

            results, crit_edges, crit_nodes = run_complete_pipeline_with_visuals(G, max_dimension=2, seed=42)
            # crit = identify_critical_cells_from_attrs(G, faces = None, node_attr = "f", edge_attr = "f")
            # crit_nodes = {cell for (cell, _, dim) in crit if dim == 0}
            # crit_edges = {canon_edge(*cell) for (cell, _, dim) in crit if dim == 1}

            # f_attr = make_attr_morse_function(G, node_attr = "f", edge_attr = "f", face_values = None)
            # cover_walk = critical_covering_walk_high_to_low(G, crit, f_attr)
            cover_walk = get_critical_path(results)

            true_walk_edges = set()

            for cell in cover_walk:
                if isinstance(cell, tuple) and len(cell) == 2:
                    u, v = int(cell[0]), int(cell[1])
                    true_walk_edges.add(canon_edge(u, v))

            true_walk_edges = sorted(true_walk_edges)

            true_crit_edges = list(crit_edges)
            true_crit_nodes = list(crit_nodes)

            if self.opt.morse_control == "morse":
                selected_crit_edges = true_crit_edges
                selected_crit_nodes = true_crit_nodes
                selected_walk_edges = true_walk_edges

            elif self.opt.morse_control == "random":
                (
                    selected_crit_edges,
                    selected_crit_nodes,
                    selected_walk_edges,
                ) = sample_random_control(
                    adj=adj,
                    num_critical_edges=len(true_crit_edges),
                    num_critical_nodes=len(true_crit_nodes),
                    num_walk_edges=len(true_walk_edges),
                    seed=self.opt.morse_seed,
                )

            else:
                raise ValueError(
                    f"Unknown Morse control: {self.opt.morse_control}"
                )

            # Critical-node indicator
            input_crit = torch.zeros(
                (num_nodes, 1),
                dtype=torch.float32,
            )

            for v in selected_crit_nodes:
                input_crit[int(v), 0] = 1.0

            self.input_crit = (
                input_crit
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(self.device)
            )

            # Critical-edge adjacency
            Edge_crit = torch.zeros(
                (num_nodes, num_nodes),
                dtype=torch.float32,
            )

            for u, v in selected_crit_edges:
                u, v = int(u), int(v)
                Edge_crit[u, v] = 1.0
                Edge_crit[v, u] = 1.0

            self.Edge_crit = Edge_crit.to(self.device)

            # Skeleton/walk adjacency
            adj_walk = torch.zeros(
                (num_nodes, num_nodes),
                dtype=torch.float32,
                device=self.device,
            )

            for u, v in selected_walk_edges:
                u, v = int(u), int(v)
                adj_walk[u, v] = 1.0
                adj_walk[v, u] = 1.0

        
            # spatial and temporal positional embeddings
            # adj = input['adj'][0]
            adj_max = torch.maximum(adj, adj.t())
            # self.adj_mask = (adj_max.mm(adj_max) == 0).to(self.device)  # mask for spatial transformer, 2-hop neighbors
            ## Laplacian eigenvectors as Positional Encodings (PE)
            ## https://arxiv.org/pdf/2003.00982.pdf
            self.pos_enc = laplacian_positional_encoding(adj_max, self.pos_dim)
            self.tpe = torch.from_numpy(temporal_positional_embedding(8, self.pos_dim)).float()
            # adjacency matrix
            # self.adj = norm_adj(adj.to(self.device)) # [[N, N], [N, N]]
            self.adj = adj.to(self.device)
            self.t_his = self.opt.t_len // 2
            assert self.t_his == self.opt.t_len - self.t_his

            adj_walk_max = torch.maximum(adj_walk, adj_walk.t()).detach().cpu().numpy()
            adj_walk_np = adj_walk.detach().cpu().numpy()

            self.walk_enc = laplacian_positional_encoding(
                adj_walk_np,
                self.pos_dim,
            )

            print(
                "[Morse control]",
                {
                    "mode": self.opt.morse_control,
                    "morse_seed": self.opt.morse_seed,
                    "num_critical_nodes": len(selected_crit_nodes),
                    "num_critical_edges": len(selected_crit_edges),
                    "num_walk_edges": len(selected_walk_edges),
                },
            )
        ####################

        if self.opt.phase == 'train':
            # random flip
            sign_flip = np.random.rand(self.pos_enc.shape[1])
            sign_flip = np.where(sign_flip > 0.5, 1, -1)
            spe = self.pos_enc * sign_flip[np.newaxis, :]
            wae = self.walk_enc * sign_flip[np.newaxis, :]
        else:
            spe = self.pos_enc
            wae = self.walk_enc
            

        self.side_info = {}
        self.side_info['covariate'] = input['feat'].to(self.device) if 'feat' in input.keys() else None
        self.side_info['spe'] = torch.from_numpy(spe).to(self.device).float() # [N, D]
        self.side_info['tpe'] = self.tpe.to(self.device) # [L, D]
        self.side_info['wae'] = torch.from_numpy(wae).to(self.device).float()
        # self.side_info['adj_mask'] = self.adj_mask


    def forward(self, training=True):
        num_batch = self.pred_gt.shape[0]
        # context encoding
        if hasattr(self, 'covariate'):
            encoder_input = torch.cat([self.pred_gt[:, :, :self.t_his], self.covariate[:, :, :self.t_his]], dim=-1)
        else:
            encoder_input = self.pred_gt[:, :, :self.t_his]

        # alpha = torch.sigmoid(self.edge_crit_gate)
        # weighted_adj = norm_adj(self.adj + alpha * self.Edge_crit) # [N, N]    
        # historical_encoding, _, = self.netEncoder(encoder_input, weighted_adj, mask_node=False)

        # historical_encoding, _, = self.netEncoder(encoder_input, self.adj, mask_node=False)

        historical_encoding, _, = self.netEncoder(encoder_input, self.adj, self.Edge_crit, mask_node=False)

        # Edge_noncrit_mask_path = os.path.join(self.opt.checkpoints_dir, self.opt.pretrain, f"Edge_noncrit_mask_N{self.pred_gt.shape[1]}.pt")

        # Edge_noncrit_mask_path = os.path.join("checkpoints/PEMS03/gwavenet_NA_20260110T090301", f"Edge_noncrit_mask_N{self.pred_gt.shape[1]}.pt")
        # self.Edge_noncrit_mask_path = Edge_noncrit_mask_path
        # historical_encoding, _, = self.netEncoder(encoder_input, self.adj, self.Edge_crit, self.Edge_noncrit_mask_path, mask_node=False)
        
        if not training:
            self.pred = self.ddim_forecasting(historical_encoding, deterministic=True)
            if self.opt.phase == 'test':
                self.sampled_pred = []
                for _ in range(self.num_sample):
                    sampled_pred = self.ddim_forecasting(historical_encoding, deterministic=False)
                    self.sampled_pred.append(sampled_pred)
                self.sampled_pred = torch.stack(self.sampled_pred, dim=1)  # [B, num_sample, N, L, D]
        else:
            # training
            # diffusion step sampling
            self.future_gt = self.pred_gt[:, :, self.t_his:]  # [B, N, L, D]
            t = torch.randint(0, self.num_steps, [num_batch]).to(self.device)  # sample diffusion steps
            current_alpha = self.alpha_hat[t].unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B,1,1,1]
            self.noise = torch.randn_like(self.future_gt)
            self.side_info['diffusion_step'] = t
            noisy_data = (current_alpha ** 0.5) * self.future_gt + (1.0 - current_alpha) ** 0.5 * self.noise
            if hasattr(self, 'covariate'):
                future_covariate = self.covariate[:, :, self.t_his:] #  [B, N, L, D]
                noisy_data = torch.cat([noisy_data, future_covariate], dim=-1)
            # print(self.future_gt.shape) torch.Size([64, 358, 12, 1])
            # print(self.t_his) 12
            # print(historical_encoding.shape) torch.Size([64, 358, 4, 64])
            # print(noisy_data.shape) torch.Size([64, 358, 12, 2])
            # self.pred = self.netSTD(noisy_data, historical_encoding, self.side_info, training)
            self.pred = self.netSTD(noisy_data, historical_encoding, self.side_info, self.input_crit, training)

    def ddim_forecasting(self, condition, deterministic=False):
        # if deterministic, use the mean of the distribution else sample from it
        target_shape = self.pred_gt[:, :, self.t_his:].shape
        num_batch = self.pred_gt.shape[0]

        # diffusion steps
        current_sample = torch.randn(target_shape) if not deterministic else torch.zeros(target_shape)
        current_sample = current_sample.to(self.device)

        step_sample_list = []
        for t in range(self.num_steps-1, -1, -1):
            step_sample_list.append(current_sample)

            if hasattr(self, 'covariate'):
                current_input = torch.cat([current_sample, self.covariate[:, :, :self.t_his]], dim=-1)
            else:
                current_input = current_sample

            # target samples
            self.side_info['diffusion_step'] = torch.tensor([t]).repeat(num_batch).to(self.device)
            # prediction = self.netSTD(current_input, condition, self.side_info, training=False)
            prediction = self.netSTD(current_input, condition, self.side_info, self.input_crit, training=False)

            if self.objective == 'noise':
                coeff1 = 1 / self.alpha[t] ** 0.5
                coeff2 = (1 - self.alpha[t]) / (1 - self.alpha_hat[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * prediction)

                if t > 0:
                    noise = torch.randn_like(current_sample) if not deterministic else torch.zeros_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha_hat[t - 1]) / (1.0 - self.alpha_hat[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            elif self.objective == 'input':
                alpha_hat = self.alpha_hat[t]
                alpha_hat_prev = self.alphas_hat_prev[t]
                noise = (current_sample - alpha_hat.sqrt() * prediction) / (1 - alpha_hat).sqrt()
                current_sample = alpha_hat_prev.sqrt() * prediction + \
                                 (1 - alpha_hat_prev).sqrt() * noise
        return current_sample

    def backward(self):
        if self.objective == 'noise':
            gt = self.noise
        elif self.objective == 'input':
            gt = self.future_gt
        else:
            raise NotImplementedError

        self.loss_l2 = self.l2_loss(self.pred, gt)
        self.loss_l2.backward()
        # self.loss_crps = self.criterion(self.pred, gt)
        # self.loss_crps.backward()
        # self.loss_mae = self.criterion(self.pred, gt)
        # self.loss_mae.backward()

    def l2_loss(self, predicted, gt):
        """ Compute the loss function. """
        # predicted: [B, num_n, time, d_x]
        # gt: [B, num_n, time, d_x]
        # missing_index: [B, num_n, time] 1 means missing signals
        assert predicted.shape == gt.shape, "predicted and noise should have the same shape"
        loss = torch.sum((predicted - gt) ** 2, dim=-2).mean()
        return loss
    
    def crps_loss(self, pred, gt, mask=None):
        return crps_mc_loss(pred, gt, mask)
    
    def mae_loss(self, pred, gt, mask=None):
        """
        Mean Absolute Error (MAE)

        pred   : Tensor [B, ...]
        gt     : Tensor [B, ...]
        mask   : optional Tensor same shape, 1 for valid, 0 for ignore
        """
        error = torch.abs(pred - gt)

        if mask is not None:
            error = error * mask
            return error.sum() / mask.sum().clamp(min=1)
        else:
            return error.mean()

    def cache_results(self):
        self._add_to_cache('missing_mask', self.missing_mask[:, :, self.t_his:])
        self._add_to_cache('pred', self.pred, reverse_norm=True)
        self._add_to_cache('gt', self.pred_gt[:, :, self.t_his:], reverse_norm=True)
        if self.opt.phase == 'test':
            self._add_to_cache('sampled_pred', self.sampled_pred, reverse_norm=True)

    def compute_metrics(self):
        pred = self.results['pred']  # [B, N, L, D]
        gt = self.results['gt']  # [B, N, L, D]
        missing_mask = self.results['missing_mask']  # [B, N, L, D]
        mae_list, rmse_list, mape_list = [], [], []
        for i in range(12):
            mae_list.append(_mae_with_missing(pred[:,:,i], gt[:,:,i], missing_mask[:,:,i]))
            rmse_list.append(_rmse_with_missing(pred[:,:,i], gt[:,:,i], missing_mask[:,:,i]))
            mape_list.append(_mape_with_missing(pred[:,:,i], gt[:,:,i], missing_mask[:,:,i]))
        self.metric_MAE, self.metric_RMSE, self.metric_MAPE = np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)

        if self.opt.phase == 'test':
            sampled_pred = self.results['sampled_pred']  # [B, num_sample, N, L, D]
            crps_list = []
            for i in range(12):
                crps_list.append(_quantile_CRPS_with_missing(sampled_pred[:,:,:,i], gt[:,:,i], missing_mask[:,:,i]))
            self.metric_CRPS = np.mean(crps_list)

    def optimize_parameters(self):
        self.set_requires_grad(self.netEncoder, True)
        self.set_requires_grad([self.netSTD], True)
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        torch.nn.utils.clip_grad_norm_(self.netSTD.parameters(), 5)
        self.optimizer.step()

    def load_networks(self, epoch=None):
        """As this model contains pretrained networks, we need to load them separately.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if epoch is not None:
            name = 'STD'
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

        # load pre-trained waveNet
        if self.opt.phase != 'test':
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrain)
        else:
            load_dir = self.save_dir
        for name in ['Encoder']:
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            load_path = os.path.join(load_dir, 'best_net_%s.pth' % name)
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)