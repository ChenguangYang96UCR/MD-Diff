from .Airbase_dataset import AirDataset


class BJAirDataset(AirDataset):

    def __init__(self, opt):
        aq_location_path = 'dataset/airquality/beijing/beijing_adj.npy'
        data_path = 'dataset/airquality/beijing/beijing.h5'
        test_nodes_path = 'dataset/airquality/beijing/test_nodes.npy'

        super().__init__(opt, aq_location_path, data_path, test_nodes_path)
