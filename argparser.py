import argparse
import sys
import gpustat
import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--loss_type', default='ce', type=str, choices=['ce'])
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--optimizer_type', default='adam', type=str, choices=['sgd', 'adam', 'rmsprop', 'adagrad'])
parser.add_argument('--use_pretrained_embedding', default=False, type=bool)
parser.add_argument('--word_embedding_dim', default=64, type=int)
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1, the use CPU.')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--train_proportion', default=0.8, type=float)
parser.add_argument('--save_checkpoint', default=False, type=bool)
parser.add_argument('--save_dir', default="saved_models/fruad", type=str)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--dataset', default="fake-news", type=str)


args = parser.parse_args()
args.datapath = "../data/%s.csv" % args.dataset
args.datadir = "../data"

if args.save_dir is None:
    args.save_dir = 'saved_models/%s' % args.network

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
print(args)

# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB
def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return gpus[np.argmin(mem)]

CPU = torch.device('cpu')
PRETRAIN_DEVICE = torch.device('cuda:4')
TRAIN_DEVICE = torch.device('cuda:5')
if torch.cuda.is_available() and args.gpu != -2:
    if args.gpu < 0:
        args.gpu = select_free_gpu()
    DEVICE = torch.device('cuda:' + str(args.gpu))
    print('use gpu indexed: %d' % args.gpu)
else:
    DEVICE = torch.device('cpu')
    print('use cpu')
