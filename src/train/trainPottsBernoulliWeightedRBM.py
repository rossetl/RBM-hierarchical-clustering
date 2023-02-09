#!/usr/bin/python3

import sys
import os
if os.getenv('HOME_PATH') != None:
    os.chdir(os.getenv('HOME_PATH'))
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/src/RBMs')
import importlib
import argparse
import utilities
import torch
from torch.utils.data import Dataset
import numpy as np
from h5py import File

# dataset class for this RBM

class RBMdataset(Dataset):
    def __init__(self, file_path, dataset='train'):
        f = File(file_path, 'r')
        self.file_path = file_path
        self.data = torch.tensor(f[dataset][()]).type(torch.int64)
        self.weights = torch.tensor(f[dataset + '_weights'][()]).type(torch.float32)
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        weight = self.weights[idx]
        return sample, weight
    
    def get_num_visibles(self):
        return self.data.shape[1]
    
    def get_dataset_mean(self):
        return (self.data.type(torch.float32) * self.weights.reshape(self.data.shape[0], 1)).sum(0) / self.weights.sum()
    
def initVisBias(dataset : torch.utils.data.Dataset) -> torch.Tensor:
    """Initialize the visible biases by minimizing the distance with the independent model.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        device (torch.device, optional): Device. Defaults to torch.device('cpu').
        dtype (torch.dtype, optional): Data type. Defaults to torch.float.

    Returns:
        torch.Tensor: Visible biases of the independent variables model.
    """
    
    X = dataset.data
    w = dataset.weights
    eps = 1e-7
    Ns = X.shape[0]
    num_states = torch.max(X) + 1
    all_states = torch.arange(num_states).reshape(-1,1,1)
    
    freq = ((X == all_states) * w.reshape(1, Ns, 1)).sum(1).type(torch.float32) / w.sum()
    freq = torch.clamp(freq, min=eps, max=1 - eps)
    
    return torch.log(freq) - 1/num_states * torch.sum(torch.log(freq), 0)
                          
# import command-line input arguments
parser = argparse.ArgumentParser(description='Train an RBM with Potts variables in the visible layer and binary variables in the hidden layer. The training is weighted.')
parser.add_argument('--train_mode',         type=str,   default='new',      help='(Defaults to new).Wheather to start a new training or recover a new one.', choices=['new', 'restore'])
parser.add_argument('--num_states',         type=int,   default=21,         help='(Defaults to 21). Number of states of the Potts variables.')
parser.add_argument('--train_type',         type=str,   default='PCD',      help='(Defaults to PCD). How to perform the training.', choices=['PCD', 'CD', 'Rdm'])
parser.add_argument('--N_save',             type=int,   default=500,        help='(Defaults to 500). Number of models to save during the training.')
parser.add_argument('--ep_max',             type=int,   default=10000,      help='(Defaults to 10000). Number of epochs.')
parser.add_argument('--Nh',                 type=int,   default=512,        help='(Defaults to 512). Number of hidden units.')
parser.add_argument('--lr',                 type=float, default=0.001,      help='(Defaults to 0.001). Learning rate.')
parser.add_argument('--NGibbs',             type=int,   default=100,        help='(Defaults to 100). Number of Gibbs steps for each gradient estimation.')
parser.add_argument('--n_mb',               type=int,   default=128,        help='(Defaults to 128). Minibatch size.')
parser.add_argument('--n_pcd',              type=int,   default=128,        help='(Defaults to 128). Number of permanant chains.')
parser.add_argument('--spacing',            type=str,   default='linear',   help='(Defaults to linear). Spacing to save models.', choices=['exp', 'linear'])
parser.add_argument('--center_gradient',    type=bool,  default=True,       help='(Defaults to True). Wheather to use the centred gradient or not.')
parser.add_argument('--seed',               type=int,   default=0,          help='(Defaults to 0). Random seed.')
args = parser.parse_args()
                    
# import the proper RBM class
RBM = importlib.import_module('PottsBernoulliWeightedRBM').RBM

device = utilities.select_device()

#####################################################################################################

# start a new training
if args.train_mode == 'new':

    # initialize random states
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import data and RBM model
    fname_data = utilities.catch_file(main_repo='data', message='Insert the filename of the dataset: ')
    train_dataset = RBMdataset(fname_data, dataset='train')
    Nv = train_dataset.get_num_visibles()
    rbm = RBM(num_visible=Nv, num_hidden=args.Nh, num_states=args.num_states, device=device)
    rbm.dataset_filename = fname_data

    fname_out = rbm.generate_model_stamp(dataset_stamp=fname_data.split('/')[-1][:-3],
                                           epochs=args.ep_max,
                                           learning_rate=args.lr,
                                           gibbs_steps=args.NGibbs,
                                           batch_size=args.n_mb,
                                           training_mode=args.train_type)
    
    model_folder = 'models/' + fname_data.split('/')[-1].split('.')[0]
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    rbm.file_stamp = fname_out

    # Initialize the visible biases.
    rbm.vbias = initVisBias(train_dataset).to(device)
    
    # Check if the file already exists. If so, ask wheather to overwrite it or not.
    utilities.check_if_exists(fname_out)

    # Select the list of training times (ages) at which saving the model.
    if args.spacing == 'exp':
        list_save_rbm = []
        xi = args.ep_max - 1
        for i in range(args.N_save):
            list_save_rbm.append(xi)
            xi = xi / args.ep_max ** (1 / args.N_save)

        list_save_rbm = np.unique(np.array(list_save_rbm, dtype=np.int32))
    
    elif args.spacing== 'linear':
        list_save_rbm = np.linspace(1, args.ep_max, args.N_save).astype(np.int32)

    else:
        sys.exit('The selected saving model interval does not exists')

    rbm.list_save_rbm = list_save_rbm

    # fit the model
    rbm.fit(train_dataset,
              training_mode=args.train_type,
              epochs=args.ep_max,
              num_pcd=args.n_pcd,
              lr=args.lr,
              batch_size=args.n_mb,
              gibbs_steps=args.NGibbs,
              updCentered=args.center_gradient
             )
    
    print('\nTraining time: {:.1f} minutes'.format(rbm.training_time / 60))

#################################################################################################

# restore an old training session
elif args.train_mode == 'restore':

    model_path = utilities.catch_file('models', message='Insert the path of the existing model: ')
    f_rbm = File(model_path, 'a')

    # restore the random states
    torch.set_rng_state(torch.tensor(np.array(f_rbm['torch_rng_state'])))
    np_rng_state = tuple([f_rbm['numpy_rng_arg0'][()].decode('utf-8'),
                            f_rbm['numpy_rng_arg1'][()],
                            f_rbm['numpy_rng_arg2'][()],
                            f_rbm['numpy_rng_arg3'][()],
                            f_rbm['numpy_rng_arg4'][()]])
    np.random.set_state(np_rng_state)
    
    # load RBM
    rbm = RBM(num_visible=0, num_hidden=0, device=device)
    rbm.loadRBM(model_path)

    # load data
    f_data = File(rbm.dataset_filename, 'r')
    train_dataset = RBMdataset(rbm.dataset_filename, dataset='train')
    f_data.close()

    ep_max_old = rbm.ep_tot
    print('This model has been trained until age: ', ep_max_old, 'epochs')
    ep_max_new = int(input('Insert the new age (epochs): ')) # put the same t_age of the previous RBM to recover an interrupted training

    rbm.ep_start = ep_max_old + 1
    rbm.ep_max = ep_max_new
    
    if ep_max_new > rbm.list_save_rbm[-1]:
        if args.spacing == 'exp':
            dlog = np.log(rbm.list_save_rbm[-1] / rbm.list_save_rbm[-2]) # previous log spacing
            new_saving_points = np.exp(np.arange(np.log(ep_max_old) + dlog, np.log(ep_max_new), dlog)).astype(np.int64)
            new_saving_points = np.append(new_saving_points - 1, ep_max_new)
        elif args.spacing == 'linear':
            dlin = rbm.list_save_rbm[-1] - rbm.list_save_rbm[-2]
            new_saving_points = np.arange(ep_max_old + dlin, ep_max_new, dlin).astype(np.int64)
            new_saving_points = np.append(new_saving_points, [ep_max_new])
        rbm.list_save_rbm = np.append(rbm.list_save_rbm, new_saving_points)
    
    rbm.fit(train_dataset, restore=True, batch_size=args.n_mb)
    f_rbm.close()
    print('\nTraining time: {:.1f} minutes'.format(rbm.training_time / 60))