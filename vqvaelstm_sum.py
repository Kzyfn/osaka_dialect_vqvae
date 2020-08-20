

from os.path import expanduser, join
import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='LSTM VAE', )
    parser.add_argument(
        '-ne',
        '--num_epoch',
        type=int,
        default=30,
    )
    parser.add_argument(
        '-nl',
        '--num_lstm_layers',
        type=int,
        #required=True,
        default=1,
    )
    parser.add_argument(
        '-zd',
        '--z_dim',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-od',
        '--output_dir',
        type=str,
        required=True,
    ),
    parser.add_argument(
        '-dr',
        '--dropout_ratio',
        type=float,
        default=0.3,
    ),
    parser.add_argument(
        '-mp',
        '--model_path',
        type=str,
        default='',
    ),
    parser.add_argument(
        '-nc',
        '--num_class',
        type=int,
        default=2,
    ),
    parser.add_argument(
        '-tr',
        '--train_ratio',
        type=float,
        default=1.0
    )

    return parser.parse_args()

args = parse()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)



import sys

import time

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset#これはなに
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
import librosa
import librosa.display




DATA_ROOT = "./data/basic5000"#NIT-ATR503/"#
test_size = 0.01 # This means 480 utterances for training data
random_state = 1234



mgc_dim = 180#メルケプストラム次数　？？
lf0_dim = 3#対数fo　？？ なんで次元が３？
vuv_dim = 1#無声or 有声フラグ　？？
bap_dim = 15#発話ごと非周期成分　？？

duration_linguistic_dim = 438#question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442#上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim #aoustice modelで求めたいもの

fs = 48000
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
hop_length = int(0.001 * frame_period * fs)

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

use_phone_alignment = True
acoustic_subphone_features = "coarse_coding" if use_phone_alignment else "full" #とは？


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))
 
def calc_lf0_rmse(natural, generated, lf0_idx, vuv_idx):
    idx = (natural[:, vuv_idx] * (generated[:, vuv_idx] >= 0.5)).astype(bool)
    return rmse(natural[idx, lf0_idx], generated[idx, lf0_idx]) * 1200 / np.log(2)  # unit: [cent]

from models import VQVAE, BinaryFileSource



X = {"acoustic": {}}
Y = {"acoustic": {}}
utt_lengths = { "acoustic": {}}
for ty in ["acoustic"]:
    for phase in ["train", "test"]:
        train = phase == "train"
        x_dim = duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
        y_dim = duration_dim if ty == "duration" else acoustic_dim
        X[ty][phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format(ty)),
                                                       dim=x_dim,
                                                       train=train))
        Y[ty][phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format(ty)),
                                                       dim=y_dim,
                                                       train=train))
        utt_lengths[ty][phase] = np.array([len(x) for x in X[ty][phase]], dtype=np.int)





X_min = {}
X_max = {}
Y_mean = {}
Y_var = {}
Y_scale = {}

for typ in ["acoustic"]:
    X_min[typ], X_max[typ] = minmax(X[typ]["train"], utt_lengths[typ]["train"])
    Y_mean[typ], Y_var[typ] = meanvar(Y[typ]["train"], utt_lengths[typ]["train"])
    Y_scale[typ] = np.sqrt(Y_var[typ])




from torch.utils import data as data_utils


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
from torch import optim
import torch.nn.functional as F


z_dim = args.z_dim
dropout= args.dropout_ratio

device = 'cuda' if torch.cuda.is_available() else 'cpu'







#model.load_state_dict(torch.load('vae_mse_0.01kld_z_changed_losssum_batchfirst_10.pth'))
# In[104]:



mora_index_lists = sorted(glob(join('data/basic5000/mora_index', "squeezed_*.csv")))
#mora_index_lists = mora_index_lists[:len(mora_index_lists)-5] # last 5 is real testset
mora_index_lists_for_model = [np.loadtxt(path).reshape(-1) for path in mora_index_lists]

train_mora_index_lists = []
test_mora_index_lists = []
#train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

for i, mora_i in enumerate(mora_index_lists_for_model):
    if (i - 1) % 20 == 0:#test
        pass
    elif i % 20 == 0:#valid
        test_mora_index_lists.append(mora_i)
    else:
        train_mora_index_lists.append(mora_i)



model = VQVAE().to(device)

if args.model_path != '':
    model.load_state_dict(torch.load(args.model_path))

optimizer = optim.Adam(model.parameters(), lr=2e-3)#1e-3

start = time.time()
beta = 0.3

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, z, z_unquantized):
    MSE = F.mse_loss(recon_x.view(-1), x.view(-1, ), reduction='sum')#F.binary_cross_entropy(recon_x.view(-1), x.view(-1, ), reduction='sum')

    with torch.no_grad():
        z_no_grad = z
        z_unquantized_no_grad = z_unquantized

    vq_loss = F.mse_loss(z.view(-1), z_unquantized_no_grad.view(-1, ), reduction='sum') + beta * F.mse_loss(z_no_grad.view(-1), z_unquantized.view(-1, ), reduction='sum')
    #print(KLD)
    return MSE +  vq_loss


func_tensor = np.vectorize(torch.from_numpy)

X_acoustic_train = [X['acoustic']['train'][i] for i in range(len(X['acoustic']['train']))] 
Y_acoustic_train = [Y['acoustic']['train'][i] for i in range(len(Y['acoustic']['train']))]
train_mora_index_lists = [train_mora_index_lists[i] for i in range(len(train_mora_index_lists))]

train_num = int(len(X_acoustic_train) * args.train_ratio)

X_acoustic_test = [X['acoustic']['test'][i] for i in range(len(X['acoustic']['test']))]
Y_acoustic_test = [Y['acoustic']['test'][i] for i in range(len(Y['acoustic']['test']))]
test_mora_index_lists = [test_mora_index_lists[i] for i in range(len(test_mora_index_lists))]

train_loader = [[X_acoustic_train[i], Y_acoustic_train[i], train_mora_index_lists[i]] for i in range(len(train_mora_index_lists))][:train_num]
test_loader = [[X_acoustic_test[i], Y_acoustic_test[i], test_mora_index_lists[i]] for i in range(len(test_mora_index_lists))]


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []

        
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).to(device))


        optimizer.zero_grad()
        recon_batch, z, z_unquantized = model(tmp[0], tmp[1], data[2])
        loss = loss_function(recon_batch, tmp[1],  z, z_unquantized )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        del tmp
        if batch_idx % 4945 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, train_num,
                100. * batch_idx / train_num,
                loss.item()))


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))
    
    return train_loss / len(train_loader)


def test(epoch):
    model.eval()
    test_loss = 0
    f0_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            tmp = []

     
            for j in range(2):
                tmp.append(torch.tensor(data[j]).to(device))


            recon_batch, z, z_unquantized = model(tmp[0], tmp[1], data[2])
            test_loss += loss_function(recon_batch, tmp[1],  z, z_unquantized).item()
            f0_loss += calc_lf0_rmse(recon_batch.cpu().numpy().reshape(-1, 199), tmp[1].cpu().numpy().reshape(-1, 199), lf0_start_idx, vuv_start_idx)
            del tmp

    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss, f0_loss






loss_list = []

test_loss_list = []

num_epochs = args.num_epoch
test_f0_losses = []
#model.load_state_dict(torch.load('vae.pth'))

for epoch in range(1, num_epochs + 1):
    loss = train(epoch)
    test_loss, test_f0_loss = test(epoch)
    print(loss)
    print(test_loss)

    print('epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
        loss,
        test_loss))

    # logging
    loss_list.append(loss)
    test_loss_list.append(test_loss)
    test_f0_losses.append(test_f0_loss)

    print(time.time() - start)

    if args.model_path != '':
        pre_trained_epoch = int(args.model_path[args.model_path.index('model_')+6:args.model_path.index('.pth')])
    else:
        pre_trained_epoch = 0

    if epoch % 5 == 0:
        torch.save(model.state_dict(), args.output_dir + '/model_'+str(epoch+pre_trained_epoch)+'.pth')
    np.save(args.output_dir +'/loss_list.npy', np.array(loss_list))
    np.save(args.output_dir +'/test_loss_list.npy', np.array(test_loss_list))
    np.save(args.output_dir +'/test_f0loss_list.npy', np.array(test_f0_losses))

