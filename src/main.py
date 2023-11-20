import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader, get_single_modal_loader
from create_dataset import OmgBehavior
def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

if __name__ == '__main__':

    args = get_args()
    dataset = 'OmgBehavior'
    bs = args.batch_size
    set_seed(args.seed)
    print("Start loading the data....")
    trainDataset, validDataset, testDataset = OmgBehavior(mode='train'), OmgBehavior(mode='valid'), OmgBehavior(mode='test')
    args.n_train, args.n_valid, args.n_test = len(trainDataset), len(validDataset), len(testDataset)
    trainDataloader, validDataloader, testDataloader = \
                            get_dataloader(args, trainDataset), get_dataloader(args, validDataset), get_dataloader(args, testDataset)

    # addintional appending
    # args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = 512, 960, 70
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = 6
    args.init_checkpoint = './t5-base/pytorch_model.bin'
    # args.init_checkpoint = './t5-large/pytorch_model.bin'


    ###adapter
    args.adapter_initializer_range = 0.001
    
    print('Finish loading the data....')
    solver = Solver(args, train_loader=trainDataloader, dev_loader=validDataloader,
                        test_loader=testDataloader, is_train=True)

    # pretrained_emb saved in train_config here

    solver.train_and_eval()