#!/usr/bin/python3

import sys
import os
if os.getenv('HOME_PATH') != None:
    os.chdir(os.getenv('HOME_PATH'))
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/src/RBMs')
import shutil
import random
import matplotlib.colors as plt_colors
import torch
import numpy as np
from h5py import File
from sklearn.cluster import DBSCAN
from ete3 import Tree
import time
import json
from tqdm import tqdm
import utilities
import importlib
import argparse

class TreeRBM():
    """
    Class that implements the construction of a tree exploiting the training hystory of an RBM.
    """
    
    def __init__(self, fname_model : str, device=torch.device('cpu')) -> None:
        """
        Args:
            fname_tree (str) : Path of the RBM model to load.
            device (torch.device) : device where to mount the TreeRBM model.
        """

        self.fname_model = fname_model
        self.device = device
        f_model = File(fname_model, 'r')
        model_identifier = f_model['model_stamp'][()].decode('utf8').split('-')[0]
        f_model.close()
        RBM = importlib.import_module(model_identifier).RBM
        self.rbm = RBM(num_visible=0, num_hidden=0, device=self.device)
        self.rbm.loadRBM(fname_model)
        self.tree_codes = []
        self.t_ages = []
        self.tree = Tree()
        self.num_labels_families = 1
        self.labels = {}       # dictionary of {label_family : all unique labels}
        self.leaves_names = []
        self.labels_dict = {}
        self.colors_dict = {}
        self.max_depth = 1
        # fit parameters
        self.eps = 0
        self.order = 0
        self.batch_size = 0
        self.alpha = 0
    

    def save_tree(self, fname : str) -> None:
        """Save the treeRBM model.

        Args:
            fname (str): Path of the folder where to save the treeRBM model
        """

        print('\nSaving the tree...\n')

        tree_dict = {
            'fname_model' : self.fname_model,
            'tree_codes' : self.tree_codes.tolist(),
            't_ages' : self.t_ages.tolist(),
            'tree' : self.tree.write(),
            'num_labels_families' : self.num_labels_families,
            'labels' : self.labels,
            'leaves_names' : self.leaves_names,
            'labels_dict' : self.labels_dict,
            'colors_dict' : self.colors_dict,
            'max_depth' : self.max_depth,
            'folder_name' : fname,
            'eps' : self.eps,
            'alpha' : self.alpha,
            'batch_size' : self.batch_size,
            'order' : self.order
        }

        tree_filename = fname + '/tree.json'
        with open(tree_filename, 'w') as f:
            json.dump(tree_dict, f)
    
    def load_tree(self, fname : str) -> None:
        """Load treeRBM model from a file.

        Args:
            fname (str): Path of the treeRBM model to be loaded.
        """

        with open(fname, 'r') as f:
            jf = json.load(f)

        self.fname_model = jf['fname_model']
        f_model = File(self.fname_model, 'r')
        model_identifier = f_model['model_stamp'][()].decode('utf8').split('-')[0]
        f_model.close()
        RBM = importlib.import_module(model_identifier).RBM
        self.rbm = RBM(num_visible=0, num_hidden=0, device=self.device)
        self.tree_codes = np.array(jf['tree_codes'])
        self.t_ages = np.array(jf['t_ages'])
        self.tree = Tree(jf['tree'], format=1)
        self.num_labels_families = jf['num_labels_families']
        self.labels = jf['labels']
        self.leaves_names = jf['leaves_names']
        self.labels_dict = jf['labels_dict']
        self.colors_dict = jf['colors_dict']
        self.max_depth = jf['max_depth']
        self.folder_name = jf['folder_name']
        self.eps = jf['eps']
        self.alpha = jf['alpha']
        self.batch_size = jf['batch_size']
        self.order = jf['order']


    def get_tree_ages(self, X_data : torch.Tensor, min_increase=0.1, alpha=1e-4) -> np.ndarray:
        """Scans all t_ages and returns those that bring an increase in the
        mean-field estimate of the number of fixed points greather than
        the proportion min_increase.

        Args:
            X_data (torch.Tensor): Data to fill the tree.
            min_increase (float, optional): Fraction of fixed points increase to consider for choosing the ages of the RBM. Defaults to 0.1.
            eps (float, optional): Convergence threshold of the algorithm. Defaults to 1e-4.

        Returns:
            np.ndarray: Array of t_ages for constructing the tree.
        """

        # Get all t_ages saved
        f_model = File(self.fname_model, 'r')
        alltime = f_model['alltime'][()]
        f_model.close()

        tree_ages = []
        prev_num_fps = 1

        pbar = tqdm(sorted(alltime), colour='red', leave=False)
        pbar.set_description('Filtering the ages')
        for stamp in pbar:
            self.rbm.loadRBM(self.fname_model, stamp=stamp)
            X_mf, _ = self.rbm.iterate_mean_field(X_data, order=1, batch_size=256, alpha=alpha, verbose=False)

            fps = torch.unique(torch.argmax(X_mf, -1), dim=0).cpu().numpy()
            num_fps = len(fps)
            if (len(tree_ages) > 0) and (num_fps == prev_num_fps):
                # keep all t_ages with the same number of fixed points for tracking the dragging of the fixed points
                tree_ages.append(stamp)
            elif (num_fps - prev_num_fps) / num_fps >= min_increase:
                tree_ages.append(stamp)
                prev_num_fps = num_fps

        return np.array(tree_ages)
    
    def fit(self, X : torch.Tensor, t_ages : np.ndarray=None, batch_size=128, min_increase=0.1, eps=1., alpha=1e-4, save_node_features : bool=False, order=2, max_iter=10000) -> None:
        """Fits the treeRBM model on the data.

        Args:
            X (torch.Tensor): Data to fill the treeRBM model.
            t_ages (np.ndarray, optional): Ages of the RBM at which compute the branches of the tree. If None, t_ages are chosen automatically. Defaults to None.
            batch_size (int, optional): Batch size, to tune based on the memory availability. Defaults to 128.
            min_increase (float, optional): Relative fixed points number that has to change for saving one age. Used only if t_ages is None. Defaults to 0.1.
            eps (float, optional): Epsilon parameter of the DBSCAN. Defaults to 1..
            alpha (float, optional): Convergence threshold of the TAP equations. Defaults to 1e-4.
            save_features_depth (int, optional): Depth below which saving the states at the tree nodes.
            order (int, optional): Order of the mean-field free energy approximation. Defaults to 2.
            max_iter (int, optional): Maximum number of TAP iterations. Defaults to 10000.
        """
        
        self.eps = eps
        self.order = order
        self.alpha = alpha
        self.batch_size = batch_size

        # get t_ages
        if np.sum(t_ages):
            self.t_ages = t_ages
        else:
            self.t_ages = self.get_tree_ages(X, min_increase=min_increase)

        # initialize the RBM
        self.rbm.loadRBM(self.fname_model, stamp=self.t_ages[-1])

        # generate tree_codes
        _, mh = self.rbm.sampleHiddens(X)
        _, mv = self.rbm.sampleVisibles(mh)
        mag_state = (mv, mh)
            
        n_data = X.shape[0]
        n_levels = len(self.t_ages)

        tree_codes = np.zeros(shape=(n_data, n_levels), dtype=np.int32)
        scan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1, metric='euclidean')
        old_fixed_points_number = np.inf
        unused_levels = n_levels
        level = n_levels - 1
        
        def evaluate_mask_pipe(mask_pipe, last_classification):
            """Function that is used to propagate the classification of the representative fixed points at a certain
            age to all the data points.
            """
            classification1 = last_classification
            for mask_matrix in reversed(mask_pipe):
                classification2 = np.zeros(mask_matrix.shape[1])
                for i, mask_row in enumerate(mask_matrix):
                    classification2[mask_row] = classification1[i]
                classification1 = classification2

            return classification1
        
        mask_pipe = []
        levels_temp = []
        labels_temp = []
        fps_temp = []

        pbar = tqdm(total=len(self.t_ages), colour='red', leave=False, dynamic_ncols=True)
        pbar.set_description('Generating tree codes')
        for t_age in reversed(self.t_ages):
            pbar.update(1)

            # load the rbm
            self.rbm.loadRBM(self.fname_model, stamp=t_age)

            # Iterate TAP equations until convergence
            n = len(mag_state[0])
            mag_state = self.rbm.iterate_mean_field(mag_state, order=order, batch_size=batch_size, alpha=alpha, max_iter=max_iter, tree_mode=True, verbose=False)

            # Clustering with DBSCAN
            scan.fit(mag_state[1].cpu())
            unique_labels = np.unique(scan.labels_)
            new_fixed_points_number = len(unique_labels)
            
            # select only a representative for each cluster and propagate the new classification up to the first layer
            mask_matrix = np.ndarray((0, n))
            representative_list = [[], []]
            for l in unique_labels:
                mask = (l == scan.labels_)
                for i, mag in enumerate(mag_state):
                    representative_list[i].append(mag[mask][0].unsqueeze(0))
                mask_matrix = np.append(mask_matrix, [mask], axis=0)
            for i in range(len(representative_list)):
                representative_list[i] = torch.cat(representative_list[i], dim=0).to(self.device)
            mask_pipe.append(mask_matrix.astype(np.bool8))
            mag_state = representative_list
            level_classification = evaluate_mask_pipe(mask_pipe, unique_labels)

            # add the new classification only if the number of TAP fixed points has decreased
            if new_fixed_points_number < old_fixed_points_number:
                tree_codes[:, level] = level_classification
                unused_levels -= 1
                old_fixed_points_number = new_fixed_points_number
                
                if save_node_features:
                    if 'Potts' in self.rbm.model_stamp:
                        for lab, fp in zip(unique_labels, mag_state[0]):
                            labels_temp.append(lab)
                            levels_temp.append(level)
                            fps_temp.append(fp[:, 1].cpu().numpy())
                    else:
                        for lab, fp in zip(unique_labels, mag_state[0]):
                            labels_temp.append(lab)
                            levels_temp.append(level)
                            fps_temp.append(fp.cpu().numpy())
                level -= 1       
            
        pbar.close()
        # Subtract from the level index the number of unused levels
        levels_temp = np.array(levels_temp) - unused_levels
        # Construct the node features dictionary
        node_features_dict = {f'level{level}-label{lab}' : fp for level, lab, fp in zip(levels_temp, labels_temp, fps_temp)}

        tree_codes = tree_codes[:, unused_levels:]
        self.tree_codes = tree_codes
        self.max_depth = tree_codes.shape[1]
        
        return node_features_dict
    
    def fit_suboptimal(self, X : torch.Tensor, t_ages : np.ndarray=None, batch_size=128, min_increase=0.1, eps=1., alpha=1e-6, order=2) -> None:
        """Fits the treeRBM model on the data. Instead of aggregating progressively the data into their fixed points, at each age uses the actual data
        as initial condition for the TAP equations.

        Args:
            X (torch.Tensor): Data to fill the treeRBM model.
            t_ages (np.ndarray, optional): Ages of the RBM at which compute the branches of the tree. If None, t_ages are chosen automatically. Defaults to None.
            batch_size (int, optional): Batch size, to tune based on the memory availability. Defaults to 128.
            min_increase (float, optional): Regulates the number of models to be used. Defaults to 0.1.
            eps (float, optional): Epsilon parameter of the DBSCAN. Defaults to 1..
            alpha (float, optional): Convergence threshold of the TAP equations. Defaults to 1e-6.
            order (int, optional): Order of the mean-field free energy approximation. Defaults to 2.
        """
        
        self.eps = eps
        self.order = order
        self.alpha = alpha
        self.batch_size = batch_size

        # get t_ages
        if np.sum(t_ages):
            self.t_ages = t_ages
        else:
            self.t_ages = self.get_tree_ages(X, min_increase=min_increase)

        # initialize the RBM
        self.rbm.loadRBM(self.fname_model, stamp=self.t_ages[-1])

        # generate tree_codes
        _, mh = self.rbm.sampleHiddens(X)
        _, mv = self.rbm.sampleVisibles(mh)
        mag_state = (mv, mh)
        
        n_data = X.shape[0]
        n_levels = len(self.t_ages)

        tree_codes = np.zeros(shape=(n_data, n_levels), dtype=np.int32)
        scan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1, metric='euclidean')
        old_fixed_points_number = np.inf
        unused_levels = n_levels
        level = n_levels - 1
        
        fps_list = []

        pbar = tqdm(total=len(self.t_ages), colour='red', leave=False)
        pbar.set_description('Generating tree codes')
        for t_age in reversed(self.t_ages):
            pbar.update(1)

            # load the rbm
            self.rbm.loadRBM(self.fname_model, stamp=t_age)

            # Iterate TAP equations until convergence
            mag_state = self.rbm.iterate_mean_field((mv, mh), order=order, batch_size=batch_size, alpha=alpha, tree_mode=True, verbose=False)

            # Clustering with DBSCAN
            scan.fit(mag_state[1].cpu())
            unique_labels = np.unique(scan.labels_)
            new_fixed_points_number = len(unique_labels)

            # add the new classification only if the number of TAP fixed points has decreased
            if new_fixed_points_number < old_fixed_points_number:
                tree_codes[:, level] = scan.labels_
                unused_levels -= 1
                old_fixed_points_number = new_fixed_points_number    
            
        pbar.close()

        tree_codes = tree_codes[:, unused_levels:]
        self.tree_codes = tree_codes
        self.max_depth = tree_codes.shape[1]
        
        return fps_list
    
    def fit_temperature_annealing(self, X : torch.Tensor, batch_size=128, eps=1., alpha=1e-4, order=2) -> None:
        """Fits the treeRBM model on the data by annealing of the oldest model.

        Args:
            X (torch.Tensor): Data to fill the treeRBM model.
            t_ages (np.ndarray, optional): Ages of the RBM at which compute the branches of the tree. If None, t_ages are chosen automatically. Defaults to None.
            batch_size (int, optional): Batch size, to tune based on the memory availability. Defaults to 128.
            eps (float, optional): Epsilon parameter of the DBSCAN. Defaults to 1..
            alpha (float, optional): Convergence threshold of the TAP equations. Defaults to 1e-4.
            order (int, optional): Order of the mean-field free energy approximation. Defaults to 2.
        """
        
        self.eps = eps
        self.order = order
        self.alpha = alpha
        self.batch_size = batch_size

        # initialize the RBM
        self.rbm.loadRBM(self.fname_model)

        # generate tree_codes
        _, mh = self.rbm.sampleHiddens(X)
        _, mv = self.rbm.sampleVisibles(mh)
        mag_state = (mv, mh)
        n_data = X.shape[0]
        n_levels = 100

        tree_codes = np.zeros(shape=(n_data, n_levels), dtype=np.int32)
        scan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1, metric='euclidean')
        old_fixed_points_number = np.inf
        unused_levels = n_levels
        actual_t_ages = []
        level = n_levels - 1
        
        def evaluate_mask_pipe(mask_pipe, last_classification):
            classification1 = last_classification
            for mask_matrix in reversed(mask_pipe):
                classification2 = np.zeros(mask_matrix.shape[1])
                for i, mask_row in enumerate(mask_matrix):
                    classification2[mask_row] = classification1[i]
                classification1 = classification2

            return classification1
        
        mask_pipe = []
        fps_list = []
        beta_annealing = np.linspace(0.9, 1., n_levels)

        pbar = tqdm(total=len(beta_annealing), colour='red', leave=False)
        pbar.set_description('Generating tree codes')
        for beta in reversed(beta_annealing):
            pbar.update(1)

            # Iterate TAP equations until convergence
            n = len(mag_state[0])
            mag_state = self.rbm.iterate_mean_field(mag_state, order=order, batch_size=batch_size, alpha=alpha, tree_mode=True, verbose=False, beta=beta)

            # Clustering with DBSCAN
            scan.fit(mag_state[1].cpu())
            unique_labels = np.unique(scan.labels_)
            new_fixed_points_number = len(unique_labels)
            
            # select only a representative for each cluster and propagate the new classification up to the first layer
            mask_matrix = np.ndarray((0, n))
            representative_list = [[], []]
            for l in unique_labels:
                mask = (l == scan.labels_)
                for i, mag in enumerate(mag_state):
                    representative_list[i].append(mag[mask][0].unsqueeze(0))
                mask_matrix = np.append(mask_matrix, [mask], axis=0)
            for i in range(len(representative_list)):
                representative_list[i] = torch.cat(representative_list[i], dim=0).to(self.device)
            mask_pipe.append(mask_matrix.astype(np.bool8))
            mag_state = representative_list
            level_classification = evaluate_mask_pipe(mask_pipe, unique_labels)

            # add the new classification only if the number of TAP fixed points has decreased
            if new_fixed_points_number < old_fixed_points_number:
                tree_codes[:, level] = level_classification
                unused_levels -= 1
                actual_t_ages.append(beta)
                old_fixed_points_number = new_fixed_points_number
 
                level -= 1       
            
        pbar.close()

        tree_codes = tree_codes[:, unused_levels:]
        self.tree_codes = tree_codes
        self.max_depth = tree_codes.shape[1]
        self.t_ages = np.sort(np.array(actual_t_ages))
        
        return fps_list

    def generate_tree(self, leaves_names : list, labels_dict : list=None, colors_dict : list=None, depth : int=None) -> Tree:
        """Constructs an ete3.Tree objects with the previously fitted data.

        Args:
            leaves_names (list): Names to assign to the leaves of the tree.
            labels_dict (list, optional): Dictionaries of the kind {leaf_name : leaf_label} with the labels to assign to the leaves. Defaults to None.
            colors_dict (list, optional): Dictionaries with a mapping {label : colour}. Defaults to None.
            depth (int, optional): Maximum depth of the tree. If None, all levels are used. Defaults to None.

        Returns:
            ete3.Tree: Tree
        """
        
        self.leaves_names = leaves_names
        self.labels_dict = labels_dict
        self.colors_dict = colors_dict

        # Validate input arguments
        if labels_dict:
            if (type(labels_dict) != list) and (type(labels_dict) != tuple):
                labels_dict = [labels_dict]

            if colors_dict:
                if (type(colors_dict) != list) and (type(colors_dict) != tuple):
                    colors_dict = [colors_dict]
                assert(len(colors_dict) == len(labels_dict)), 'colors_dict must have the same length of labels_dict'

            self.num_labels_families = len(labels_dict)
            for label_family, ld in enumerate(labels_dict):
                all_unique_labels = np.unique(list(ld.values()))
                self.labels[label_family] = all_unique_labels

        if depth:
            assert(depth <= self.tree_codes.shape[1]), 'depth parameter should be <= than the tree depth'
            n_levels = depth
        else:
            n_levels = self.tree_codes.shape[1]

        # Initialize the tree with the proper number of initial branches
        n_start_branches = np.max(self.tree_codes[:, 0]) + 1
        init_tree = '('
        for i in range(n_start_branches):
            init_tree += 'R{0}-:1,'.format(i)
        init_tree = init_tree[:-1] + ')Root:1;'
        t = Tree(init_tree, format=1)
        for n in t.traverse():
            t.add_feature('level', 0)

        # Build the tree structure
        for level in range(2, n_levels + 1):
            tree_lvl = np.unique(self.tree_codes[:, :level], axis=0)
            for lvl in tree_lvl:
                leaf_name = 'R' + ''.join([str(aa) + '-' for aa in lvl])
                mother_name = 'R' + ''.join([str(aa) + '-' for aa in lvl[:-1]])
                M = t.search_nodes(name=mother_name)[0]
                M.add_child(name=leaf_name)
                C = M.search_nodes(name=leaf_name)[0]
                M.add_feature('level', level -1)
                C.add_feature('level', level)

        # Add all leaves to the tree
        for tree_node, leaf_name in zip(self.tree_codes, leaves_names):
            mother_name = 'R' + ''.join([str(aa) + '-' for aa in tree_node[:n_levels]])
            M = t.search_nodes(name=mother_name)[0]
            M.add_child(name=leaf_name)

        # add labels to the leaves
        if labels_dict:
            for i, ld in enumerate(labels_dict):

                if colors_dict:
                    leaves_colors = [colors_dict[i][label] for label in ld.values()]
                    # create annotation file for iTOL
                    f = open(self.folder_name + '/leaves_colours' + str(i) + '.txt', 'w')
                    f.write('DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\tLabel family ' + str(i) + '\nCOLOR\tred\n')
                    f.write('LEGEND_TITLE\tLabel family {0}\nSTRIP_WIDTH\t75'.format(i))
                    f.write('\nLEGEND_SHAPES')
                    for _ in colors_dict[i].keys():
                        f.write('\t2')
                    f.write('\nLEGEND_LABELS')
                    for l in colors_dict[i].keys():
                        l = l
                        f.write(f'\t{l}')
                    f.write('\nLEGEND_COLORS')
                    for c in colors_dict[i].values():
                        f.write(f'\t{c}')
                    f.write('\nDATA\n')
                    
                    for leaf_name, leaf_color, label in zip(ld.keys(), leaves_colors, ld.values()):
                        leaf = t.search_nodes(name=leaf_name)[0]
                        rgba_colors_norm = plt_colors.to_rgba(leaf_color, 1.)
                        rgba_colors = tuple(int(nc * 255) if i != 3 else nc for i, nc in enumerate(rgba_colors_norm))
                        f.write(leaf.name + '\trgba' + str(rgba_colors).replace(' ', '') + '\t' + str(label) + '\n')
                    f.close()

        self.tree = t
        
        # generate nw file
        self.tree.write(format=1, outfile=self.folder_name + '/tree.nw')

        return t

if __name__ == '__main__':
    
    # import command-line input arguments
    parser = argparse.ArgumentParser(description='Generates the tree associated of a specified RBM model.')
    parser.add_argument('--n_data',                 type=int,   default=500,        help='(Defaults to 500). Number of data to put in the tree.')
    parser.add_argument('--batch_size',             type=int,   default=128,        help='(Defaults to 128). Batch size.')
    parser.add_argument('--data_set',               type=str,   default='train',    help='(Defaults to train). Dataset to use..', choices=['train', 'test'])
    parser.add_argument('--filter_ages',            type=bool,  default=False,      help='(Defaults to False). Wheather to filter the ages with the naive MF equations or not.')
    parser.add_argument('--max_age',                type=int,   default=np.inf,     help='(Defaults to inf). Maximum age to consider for the tree construction.')
    parser.add_argument('--save_node_features',     type=bool,  default=False,      help='(Defaults to False). If True, saves the states corresponding to the tree nodes.')
    parser.add_argument('--max_iter',               type=int,   default=10000,      help='(Defaults to 10000). Maximum number of TAP iterations.')
    parser.add_argument('--min_increase',           type=float, default=0.1,        help='(Defaults to 0.1). Relative fixed points number that has to change for saving one age. Used only if filter_ages is True.')
    parser.add_argument('--eps',                    type=float, default=1.,         help='(Defaults to 1.). Epsilon parameter of the DBSCAN.')
    parser.add_argument('--alpha',                  type=float, default=1e-4,       help='(Defaults to 1e-4). Convergence threshold of the TAP equations.')

    args = parser.parse_args()

    # Select device and RBM model
    start = time.time()
    device = utilities.select_device()
    fname_model = utilities.catch_file(main_repo='models', message='Insert the path of the model: ')
    
    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Initialize the tree
    tree = TreeRBM(fname_model, device=device)
    
    # Load the data
    dataset_name = utilities.catch_file(main_repo='data', message='Insert the name of the data file: ').split('/')[-1]
    if 'Potts' in tree.rbm.model_stamp:
        data_type = torch.int64
        order_MF = 2
    else:
        data_type = torch.float32
        order_MF = 3
    f_data = File('data/' + dataset_name, 'r')
    all_X_data = np.array(f_data[args.data_set])
    X_data = torch.tensor(all_X_data[:args.n_data], device=device).type(data_type)

    # Fit the tree to the data
    if args.filter_ages:
        t_ages = None
    else:
        t_ages = tree.rbm.list_save_rbm[tree.rbm.list_save_rbm <= args.max_age]
    node_features_dict = tree.fit(X_data, batch_size=args.batch_size, t_ages=t_ages, save_node_features=args.save_node_features, min_increase=args.min_increase, eps=args.eps, alpha=args.alpha, max_iter=args.max_iter, order=order_MF)

    # Save the tree codes
    folder_name = 'trees/TreeRBM-' + tree.rbm.model_stamp + '-' + dataset_name[:-3] + '-Gibbs_steps' + str(tree.rbm.gibbs_steps) + '-' + args.data_set
    node_states_file_name = folder_name + '/node_features.h5'
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    tree.save_tree(folder_name)
    
    # create the file with the node states
    if args.save_node_features:
        f_nodes = File(node_states_file_name, 'w')
        for state_name, state in node_features_dict.items():
            f_nodes[state_name] = state
        f_nodes.close()
        
    stop = time.time()

    print('Process completed, elapsed time:', round((stop - start) / 60, 1), 'minutes')
