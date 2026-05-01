# implemented by p0werHu
from data.base_dataset import BaseDataset
import torch
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler

from data.data_util import calculate_normalized_laplacian


class AirDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(y_dim=1, covariate_dim=0)
        return parser

    def __init__(self, opt, aq_location_path, data_path, test_nodes_path):
        super().__init__(opt)
        """
        load data give options
        """
        self.opt = opt

        self.pred_attrs = opt.pred_attr
        assert self.pred_attrs == 'PM25', 'For other attributes, signals of several nodes are completely missing. Need to use other preprocessing methods.'
        self.time_division = {
            'train': [0.0, 0.7],
            'val': [0.7, 0.8],
            'test': [0.8, 1.0]
        }

        self.A = self.load_loc(aq_location_path)
        self.raw_data = self.load_feature(data_path, self.time_division[opt.phase])

        # get data division index
        self.opt.__dict__.update({'num_nodes': self.A.shape[0]})
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)

        # data format check
        self._data_format_check()

    def load_loc(self, aq_location_path):
        """
        Args:
        Returns:

        """
        print('Loading station locations...')
        # load air quality station locations data
        adj = np.load(aq_location_path)
        return adj

    def load_feature(self, data_path, time_division):
        raw_data = pd.read_hdf(data_path, key='data')
        covariates = pd.read_hdf(data_path, key='covariate')
        missing = pd.read_hdf(data_path, key='missing')

        print('Loading air quality features...')
        data = {'feat': [],
                'pred': [],
                'missing': [],
                'time': []}

        for id, station_aq in raw_data.groupby('station_id'):
            station_aq = station_aq.set_index("time").drop(columns=['station_id'])
            # split data into features and labels
            data['pred'].append(station_aq[self.pred_attrs].to_numpy()[np.newaxis])
            data['missing'].append(missing.loc[missing['station_id'] == id][self.pred_attrs + '_Missing'].to_numpy()[np.newaxis])
            data['feat'].append(covariates.loc[covariates['station_id'] == id].drop(columns=['station_id', 'time']).to_numpy()[np.newaxis])

        data['pred'] = np.concatenate(data['pred'], axis=0)[..., np.newaxis]
        data['missing'] = np.concatenate(data['missing'], axis=0)[..., np.newaxis]
        data['feat'] = np.concatenate(data['feat'], axis=0)

        # normalize data
        self.add_norm_info(np.mean(data['pred']), np.std(data['pred']))
        data['pred'] = (data['pred'] - self.opt.mean) / self.opt.scale

        data_length = data['feat'].shape[1]
        start_index, end_index = int(time_division[0] * data_length), int(time_division[1] * data_length)
        data['feat'] = data['feat'][:, start_index:end_index, :]
        data['missing'] = data['missing'][:, start_index:end_index, :]
        data['pred'] = data['pred'][:, start_index:end_index, :]

        data['time'] = station_aq[start_index:end_index].index.values.astype(np.datetime64)
        data['time'] = ((data['time'] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        # todo: I don't use covariate for diffusion models anymore
        data.pop('feat')
        return data