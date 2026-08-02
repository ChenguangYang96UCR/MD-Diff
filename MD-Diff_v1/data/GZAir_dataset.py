from .Airbase_dataset import AirDataset


class GZAirDataset(AirDataset):

    def __init__(self, opt):
        aq_location_path = 'dataset/airquality/guangzhou/guangzhou_adj.npy'
        data_path = 'dataset/airquality/guangzhou/guangzhou.h5'
        test_nodes_path = 'dataset/airquality/guangzhou/test_nodes.npy'

        super().__init__(opt, aq_location_path, data_path, test_nodes_path)
