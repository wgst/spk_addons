import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import *
import pytorch_lightning as pl
import torchmetrics
import os
import sys
import numpy as np
sys.path.append('spk_addons')
from spk_addons import *
import spk_addons
from spk_addons.logtrain import log_arguments
from spk_addons.hmodel import Forces#, HDiab
from spk_addons.data_custom import CustomData
from spk_addons.utils import read_param
#import model as hmodel
import argparse
import logging
from schnetpack.data import * 
####### Parser arguments
parser = argparse.ArgumentParser(description='PAiNN for hamiltonian and diabatization')
parser.add_argument('--datapath', help='Path to database file')
parser.add_argument('--batch_size', help='Specifies batch size, default = 10', default=1, type=int)
parser.add_argument('--num_train', help='Set size of trainingdata, default = 100', type=int, default=20)
parser.add_argument('--num_val', help='Set size of evaluationdata, default = 100', type=int, default=20)
parser.add_argument('--cutoff', help='Set cutoff radius, default = 5', type=float, default=5.)
parser.add_argument('--features', help='Number of features, default = 30', type=int, default=30)
parser.add_argument('--rootdir', help='Set path to output directory')
parser.add_argument('--split_file', help='Set path to split file, default "split.npz"', default="split.npz")
parser.add_argument('--max_epochs', help='Number of maximum epochs, default = 100', type=int, default=100)
parser.add_argument('--interactions', help='Number of interactions, default = 3', type=int, default=3)
parser.add_argument('--gpu' , help="If set, number of GPUs", type=int)
parser.add_argument('--layer' , help="Set number of layers, default = 3", type=int, default=3)
parser.add_argument('--schnet', help="Use representation SchNet", action="store_true")
parser.add_argument('--painn', help="Use representation PAiNN", action="store_true")
parser.add_argument('--lr', help="Set learning rate, default = 1e-4", type=float, default=1e-4)
parser.add_argument('--num_worker', help="Increase the number of workers to improve performance, default = 8", type=int, default=8)
parser.add_argument('--real_socs', help="Makes use of the fact that no imaginary soc values are given.",action="store_true")
parser.add_argument('--aggregation_mode', help='Aggregation mode, sum or avg, default = sum', type=str, default="sum")
parser.add_argument('--hess', help="Computes Hessian of potential energies", action="store_true")
parser.add_argument('--n_gaussian', help="Set number of radial basis functions, default = 50", type=int, default=50)
parser.add_argument('--lr_patience', help="Set learning rate patience, default = 15", type=int, default=15)
parser.add_argument('--lr_decay', help="Set learning rate decay, default = 0.8", type=int, default=0.8)
parser.add_argument('--lr_min', help="Set minimum learning rate, default = 1e-6", type=float, default=1e-6)
parser.add_argument('--comment',help="Write a comment that will be saved in the training log",type=str,default="")
parser.add_argument('--overwrite', help="Overwrite training directory", action="store_true")
parser.add_argument('--debug', help="Set logging to debug", action="store_true")
parser.add_argument('--environment', help="Set environment provider. torch or ase. torch supports PBC", type=str, default="ase")
args = parser.parse_args()
if args.overwrite:
    os.system(" rm -f %s/evaluation.txt"%args.rootdir)
split = np.load("%s"%os.path.join(args.rootdir,"split.npz"))

train_idx = split["train_idx"]
test_idx = split["test_idx"]
val_idx = split["val_idx"]

params = read_param(os.path.join(args.rootdir,"args.json"))

if params["environment"] == "ase":
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=float(params["cutoff"])))
    dataset = CustomData(args.datapath,batch_size=args.batch_size,
    num_train=int(params["num_train"]),
    num_val=int(params["num_val"]),
    transforms=[
        trn.ASENeighborList(cutoff=float(params["cutoff"])),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False, is_extensive=False),
        trn.CastTo32()
    ],
    num_workers=args.num_worker,
    split_file=os.path.join(args.rootdir, "split.npz"),
    load_properties=["energy","forces"],
    pin_memory=True, # set to false, when not using a GPU
    )
if params["environment"] == "torch":
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.TorchNeighborList(cutoff=params["cutoff"]))
    dataset = CustomData(args.datapath,batch_size=args.batch_size,
    num_train=int(params["num_train"]),
    num_val=int(params["num_val"]),
    transforms=[
        trn.TorchNeighborList(cutoff=float(params["cutoff"])),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False, is_extensive=False),
        trn.CastTo32()
    ],
    num_workers=args.num_worker,
    split_file=os.path.join(args.rootdir, "split.npz"),
    load_properties=["energy","forces"],
    pin_memory=True, # set to false, when not using a GPU
    )
dataset.prepare_data()
dataset.setup()

# Load model
best_model = torch.load(os.path.join(args.rootdir,'best_inference_model')).to("cpu")

results = {}
targets = {}
for batch in dataset.test_dataloader():
    result = best_model(batch)
    for key in result:
        if key not in results:
            results[key]=[]
        if key not in targets:
            targets[key]=[]
        results[key].append(result[key])
        targets[key].append(batch[key])
# save predictions and reference values

np.savez(args.rootdir+"/predictions.npz",results)
np.savez(args.rootdir+"/reference.npz",targets)
