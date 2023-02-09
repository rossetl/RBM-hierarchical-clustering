# RBM-hierarchical-clustering
Code for the paper "Unsupervised hierarchical clustering using the learning dynamics of RBMs" ([arXiv:2302.01851](https://arxiv.org/abs/2302.01851)) by Aurélien Decelle, Lorenzo Rosset and Beatriz Seoane.

## Installation
- Create the conda environment with all the dependences: 
```
conda env create -f RBMenv.yml
```

- Include the main directory to your PATH environment variable by adding the following line to your ~/.bashrc file:
```
export PATH=${PATH}:/path/to/RBM-hierarchical-clustering
```

## Usage

### Train an RBM model
The possible RBM models are denoted as:
- `BernoulliBernoulliRBM`: Bernoulli variables in both the visible and the hidden layer;
- `BernoulliBernoulliWeightedRBM`: Bernoulli variables in both the visible and the hidden layer. During the training, the averages over the data points are weighted;
- `PottsBernoulliRBM`: Potts variables in the visible layer and Bernoulli variables in the hidden layer;
- `PottsBernoulliWeightedRBM`: Potts variables in the visible layer and Bernoulli variables in the hidden layer. During the training, the averages over the data points are weighted.

To train a specific RBM model launch `rbm-train` followed by the proper arguments:
- *default*: trains a `BernoulliBernoulliRBM` model;
- `-V`: selects the visible layer variables type. Followed by `Bernoulli` trains a `BernoulliBernoulliRBM` model, while followed by `Potts` trains a `PottsBernoulliRBM` model;
- `-w`: trains the weighted version of the previous models;
- To open the help page type `rbm-train -h`.

Apart from the previous parameters, another series of parameters specifies the training specifics. To list all the possibilities, use the argument `-i` (e.g. `rbm-train -V -w -i`).

The script will ask to select the data file to be used for the training among those present in the folder `data/` (for details on the training data file format see the next section).

#### Data source format
The files used for the training must have the format `.h5` and must be present in the folder `data/`. The following keyword-data pairs must be present in the file:

Mandatory:
- *train*: training set data;

Optional information:
- *train_labels* : labels for the training data in string format. If no label is available for certain data points, use the dummy label '-1';
- *weights*: necessary only for the weighted versions of the training.
- *test*: test set data;
- *test_labels* : labels for the test data in string format. If no label is available for certain data points, use the dummy label '-1';

## TreeRBM
Once you have a trained model in the folder `models/`, you can use the `rbm-maketree` command to generate the hierarchical tree for a dataset compatible with the one used for the training of the RBM. Use `rbm-maketree -h` to list all the optional arguments.

The script will output a repository in the `trees/` folder with a name referring to the model used for generating the tree. Inside the repository, there will be a file `tree.json` containing the information about the TreeRBM object created and (optionally) a `node_features.h5` file containing the features at each node of the tree.

To generate a newick file of the tree and the corresponding iTOL annotation files, the TreeRBM method `generate_tree` has to be called with a proper list of arguments. See the example notebook for the usage.

## Example data
To download an example of an input file for training the RBM using the MNIST dataset, use
```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XiP_KPKuGZmxoqQz6tnVqUFlxf44S5kX' -O 'data/MNIST.h5'
```

To download an example of RBM model trained on the MNIST dataset, use
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=194iINKzWGojGr1IHhFvWwOO0ytgLlV10' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=194iINKzWGojGr1IHhFvWwOO0ytgLlV10" -O 'models/MNIST/PottsBernoulliRBM-2023.2.7.15.27-MNIST-ep10000-lr0.001-Nh512-NGibbs100-mbs500-PCD.h5' && rm -rf /tmp/cookies.txt
```

To download the corresponding TreeRBM files, use
```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1t7CR0qApAE8gC8YuwyeBZ6UZT4neogZT' -O 'trees/TreeRBM-PottsBernoulliRBM-2023.2.7.15.27-MNIST-Gibbs_steps100-train.tar.xz'
```
```
cd trees && tar -xf TreeRBM-PottsBernoulliRBM-2023.2.7.15.27-MNIST-Gibbs_steps100-train.tar.xz && rm TreeRBM-PottsBernoulliRBM-2023.2.7.15.27-MNIST-Gibbs_steps100-train.tar.xz && cd ..
```

<image src="/images/tree-MNIST.png"/>