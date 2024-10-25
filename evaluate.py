from model import *
import main
import torch
import dataset
import torch.nn as nn
import train
import utils
import torch.nn.functional as F
'''
Namespace(dataset='mLFR500_mu0.2', diffusion='ppr', initial_feat='deepwalk',
          num_layer=4, nb_epochs=101, hid_units=512, lr=0.001, l2_coef=0.0, patience=20,
          sparse=False, sample_size=500, batch_size=1, student_t_v=1, num_cluster=3, with_gt=True,
          test_Q=True, perEpoch_Q=10)
'''

import argparse
args, unknown = main.parse_args()

if 'mLFR' in args.dataset:
    args.num_layer = 4
    args.num_cluster = 3
    args.sample_size = 500
    args.with_gt = True
if 'aucs' in args.dataset:
    args.num_layer = 5
    args.num_cluster = 6
    args.sample_size = 61
    args.with_gt = False
if 'academia' in args.dataset:
    args.num_layer = 2
    args.num_cluster = 4
    args.sample_size = 500
    args.with_gt = True
    args.test_Q = False
    args.perEpoch_Q = 10
    args.weights_path = "/content/model_weights.pth"

print(args)

WEIGHTS_PATH = args.weights_path

num_layer=args.num_layer
nb_epochs = args.nb_epochs
patience = args.patience
lr = args.lr
l2_coef = args.l2_coef
hid_units = args.hid_units
sparse = args.sparse
dataset_pth = args.dataset

# Load data 
_, _, features, labels  = dataset.load(dataset_pth,1,args) # labels not used
ft_size = features.shape[1]
nb_classes = np.unique(labels).shape[0] # Not used

sample_size = args.sample_size
batch_size = args.batch_size
labels = torch.LongTensor(labels) # Not used
lbl_1 = torch.ones(batch_size, sample_size * 2)
lbl_2 = torch.zeros(batch_size, sample_size * 2)
lbl = torch.cat((lbl_1, lbl_2), 1)
model = Model(args, ft_size, hid_units)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
model.cpu()
labels = labels.cpu() # Not used
lbl = lbl.cpu() # Not used

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
data=[]

for layer in range(num_layer):
            
    adj, diff, features, labels = dataset.load(dataset_pth,layer+1,args)
    ba, bd, bf = [], [], []
    i=0
    # adj
    ba.append(adj[i: i + sample_size, i: i + sample_size])
    # diffuse
    bd.append(diff[i: i + sample_size, i: i + sample_size])
    # feature
    bf.append(features[i: i + sample_size])

    ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
    bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
    bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

    if sparse:
        ba = utils.sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
        bd = utils.sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
    else:
        ba = torch.FloatTensor(ba)
        bd = torch.FloatTensor(bd)

    bf = torch.FloatTensor(bf)
    idx = np.random.permutation(sample_size)
    shuf_fts = bf[:, idx, :]

    bf = bf.cpu()
    ba = ba.cpu()
    bd = bd.cpu()
    shuf_fts = shuf_fts.cpu()

    data.append([bf, shuf_fts, ba, bd])

    optimiser.zero_grad()

    logits, logit_all, z, q = model(data, sparse, None) 
    p = train.target_distribution(q)


    probabilities = F.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    print(predictions)