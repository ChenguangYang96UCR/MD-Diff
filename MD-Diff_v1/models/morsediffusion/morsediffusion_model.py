from models import init_net, BaseModel
import torch
import torch.nn.functional as F
from .stformer import STFormer
from utils.util import _mae_with_missing,_rmse_with_missing, _mape_with_missing, _quantile_CRPS_with_missing
from .model_util import get_schedule, laplacian_positional_encoding, temporal_positional_embedding, norm_adj
from .gwavenet import GWaveNetEncoder
import os
import numpy as np

class morsediffusionModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # modify options for the model
        return parser

    def __init__(self, opt, model_config):
        super().__init__(opt, model_config)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['l2']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['STD', 'Encoder']

        # specify metrics you want to evaluate the model. The training/test scripts will call functions in order:
        # <BaseModel.compute_metrics> compute metrics for current batch
        # <BaseModel.get_current_metrics> compute and return mean of metrics, clear evaluation cache for next evaluation
        self.metric_names = ['MAE', 'RMSE', 'MAPE']
        if self.opt.phase == 'test':
            self.metric_names += ['CRPS']

        # define networks. The model variable name should begin with 'self.net'
        model_config['task'] = 'extrapolation'
        model_config['condition_dim'] = model_config['wavenet']['end_dim']
        model_config['input_dim'] = opt.y_dim + opt.covariate_dim
        model_config['t_len'] = opt.t_len
        model_config['output_dim'] = opt.t_len  # todo: change this line
        model_config['spatial_dim'] = model_config['pos_dim']
        model_config['num_nodes'] = opt.num_nodes
        model_config['wavenet']['input_dim'] = opt.y_dim + opt.covariate_dim

        self.netEncoder = GWaveNetEncoder(model_config['wavenet'])
        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        # self.netEncoder.eval()

        self.netSTD = STFormer(model_config)
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
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.AdamW([{'params': self.netSTD.parameters()},
                                                {'params': self.netEncoder.parameters(), 'lr': opt.lr * 0.1}]
                                                ,lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)
            self.load_networks()  # only load pre-trained model

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

        #########
        #the following parts actually only run once in the training process
        #########
        if not hasattr(self, 'pos_enc'):
            # spatial and temporal positional embeddings
            adj = input['adj'][0]
            adj_max = torch.maximum(adj, adj.t())

            ## Laplacian eigenvectors as Positional Encodings (PE)
            ## https://arxiv.org/pdf/2003.00982.pdf
            self.pos_enc = laplacian_positional_encoding(adj_max, self.pos_dim)
            self.tpe = torch.from_numpy(temporal_positional_embedding(self.opt.t_len, self.pos_dim)).float()

            # side info dic contains embedding vectors for attributes (e.g., spatial, temporal, covariates, diffusion steps)
            self.context_index = input['context_index'][0].to(self.device)
            self.target_index = input['target_index'][0].to(self.device)

            # adjacency matrix
            self.context_adj = norm_adj(adj.to(self.device)[self.context_index][:, self.context_index]) # [N, N]
        ####################

        if self.opt.phase == 'train':
            # random flip
            sign_flip = np.random.rand(self.pos_enc.shape[1])
            sign_flip = np.where(sign_flip > 0.5, 1, -1)
            spe = self.pos_enc * sign_flip[np.newaxis, :]
        else:
            spe = self.pos_enc

        self.side_info = {}
        self.side_info['covariate'] = input['feat'].to(self.device) if 'feat' in input.keys() else None
        self.side_info['spe'] = torch.from_numpy(spe).to(self.device).float() # [N, D]
        self.side_info['tpe'] = self.tpe.to(self.device) # [L, D]
        self.side_info['context_index'] = self.context_index  # [N]
        self.side_info['target_index'] = self.target_index  # [N]

    def forward(self, training=True):
        num_batch = self.pred_gt.shape[0]
        # context encoding
        if hasattr(self, 'covariate'):
            encoder_input = torch.cat([self.pred_gt[:,self.context_index], self.covariate[:,self.context_index]], dim=-1)
        else:
            encoder_input = self.pred_gt[:,self.context_index]
        context_encoding, _ = self.netEncoder(encoder_input, self.context_adj, mask_node=False)

        if not training:
            self.pred = self.ddim_extrapolation(context_encoding, deterministic=True)
            if self.opt.phase == 'test':
                self.sampled_pred = []
                for _ in range(self.num_sample):
                    sampled_pred = self.ddim_extrapolation(context_encoding, deterministic=False)
                    self.sampled_pred.append(sampled_pred)
                self.sampled_pred = torch.stack(self.sampled_pred, dim=1)  # [B, num_sample, N, L, D]
        else:
            # training
            # diffusion step sampling
            self.target_gt = self.pred_gt[:, self.target_index]
            t = torch.randint(0, self.num_steps, [num_batch]).to(self.device)  # sample diffusion steps
            current_alpha = self.alpha_hat[t].unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B,1,1,1]
            self.noise = torch.randn_like(self.target_gt)
            self.side_info['diffusion_step'] = t
            noisy_data = (current_alpha ** 0.5) * self.target_gt + (1.0 - current_alpha) ** 0.5 * self.noise
            if hasattr(self, 'covariate'):
                future_covariate = self.covariate[:, self.target_index] #  [B, N, L, D]
                noisy_data = torch.cat([noisy_data, future_covariate], dim=-1)
            self.pred = self.netSTD(noisy_data, context_encoding, self.side_info, training)

    def ddim_extrapolation(self, condition, deterministic=False):
        # if deterministic, use the mean of the distribution else sample from it
        target_shape = self.pred_gt[:, self.target_index].shape
        num_batch = self.pred_gt.shape[0]

        # diffusion steps
        current_sample = torch.randn(target_shape) if not deterministic else torch.zeros(target_shape)
        current_sample = current_sample.to(self.device)

        step_sample_list = []
        for t in range(self.num_steps-1, -1, -1):
            step_sample_list.append(current_sample)

            if hasattr(self, 'covariate'):
                current_input = torch.cat([current_sample, self.covariate[:, self.target_index]], dim=-1)
            else:
                current_input = current_sample

            # target samples
            self.side_info['diffusion_step'] = torch.tensor([t]).repeat(num_batch).to(self.device)
            prediction = self.netSTD(current_input, condition, self.side_info, training=False)

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
            gt = self.target_gt
        else:
            raise NotImplementedError

        self.loss_l2 = self.l2_loss(self.pred, gt)
        self.loss_l2.backward()

    def l2_loss(self, predicted, gt):
        """ Compute the loss function. """
        # predicted: [B, num_n, time, d_x]
        # gt: [B, num_n, time, d_x]
        # missing_index: [B, num_n, time] 1 means missing signals
        assert predicted.shape == gt.shape, "predicted and noise should have the same shape"
        loss = torch.sum((predicted - gt) ** 2, dim=-2).mean()
        return loss

    def cache_results(self):
        self._add_to_cache('missing_mask', self.missing_mask[:, self.target_index])
        self._add_to_cache('pred', self.pred, reverse_norm=True)
        self._add_to_cache('gt', self.pred_gt[:, self.target_index], reverse_norm=True)
        self._add_to_cache('context_index', self.side_info['context_index'], replace=True)
        self._add_to_cache('target_index', self.side_info['target_index'], replace=True)
        if self.opt.phase == 'test':
            self._add_to_cache('sampled_pred', self.sampled_pred, reverse_norm=True)

    def compute_metrics(self):
        target_index = self.results['target_index']
        # sto_pred = self.results['sto_pred'][:, :, target_index]  # [T, num_sample, num_n]
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