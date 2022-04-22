import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import *
import pytorch_lightning as pl
import torchmetrics
import os
import sys
sys.path.append('spk_addons')
from spk_addons import *
import spk_addons
from spk_addons.utils import log_arguments
#from spk_addons.hmodel import Forces#, HDiab
from spk_addons.data_custom import CustomData
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
parser.add_argument('--rho', help='Tradeoff for trainin energies and forces, default = 0.5', type=float, default=0.5)
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
parser.add_argument('--overwrite', help="Overwrite training directory", action="store_true")
parser.add_argument('--debug', help="Set logging to debug", action="store_true")
parser.add_argument('--environment', help="Set environment provider. torch or ase. torch supports PBC", type=str, default="ase")
args = parser.parse_args()
if args.overwrite:
    os.system(" rm -rf %s"%args.rootdir)

if not os.path.exists(args.rootdir):
    os.makedirs(args.rootdir)
#Save training options

log_arguments(vars(args))
if args.debug:
    logging.basicConfig(filename='train_painn.log',level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
####### Parameter
if args.datapath:
    datapath = args.datapath
else:
    raise NameError("--datapath not specified.")
"""
import ase.db, ase.io,ase
dataset = ase.db.connect(args.datapath)
datalist=[]
print(dataset.metadata.keys())
import numpy as np
for i in range(len(dataset)):
    d=dataset.get(i+1).data
    data={}
    for key in dataset.metadata["_property_unit_dict"].keys():
        if key=="energy":
            data[key] = np.array(d[key])
        elif key == "forces":
            data[key]=np.array(d[key]).reshape(8,3)
        else:
            data[key]=np.array([d[key]])
    datalist.append(data)


molecs = ase.io.read(args.datapath,":")
p=dataset.metadata["_property_unit_dict"] 
os.system("rm -f Test.db")
db=ASEAtomsData.create("Test.db",    distance_unit=1, property_unit_dict=p)#{"energy":1,"forces":1})
db.add_systems(datalist,molecs)
"""
batch_size = args.batch_size
num_train = args.num_train
num_val = args.num_val
cutoff = args.cutoff
n_atom_basis = args.features
max_epochs = args.max_epochs
n_interactions = args.interactions
split_file = args.split_file
#Datensatz laden, prepare, etc

# get nstates by parsing metadata
n_ener = 1
if args.environment == "ase":
    dataset = CustomData(args.datapath,batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        transforms=[
        trn.ASENeighborList(cutoff=args.cutoff),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False,is_extensive=False),
        trn.CastTo32()
        ],
        num_workers=args.num_worker,
        split_file=os.path.join(args.rootdir, "split.npz"),
        load_properties=["energy","forces"],
        pin_memory=True, # set to false, when not using a GPU
    )
if args.environment == "torch":
    dataset = CustomData(args.datapath,batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        transforms=[
        trn.TorchNeighborList(cutoff=args.cutoff),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False,is_extensive=False),
        trn.CastTo32()
        ],
        num_workers=args.num_worker,
        split_file=os.path.join(args.rootdir, "split.npz"),
        load_properties=["energy","forces"],
        pin_memory=True, # set to false, when not using a GPU
    )
dataset.prepare_data()
dataset.setup()
#OAD DATA
#format = AtomsDataFormat.ASE
#dataset.setup()



pairwise_distance = spk.atomistic.PairwiseDistances() #Input for NN
#Setup representation
radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_gaussian,cutoff=cutoff)

# get representation

if args.painn:
    repr = spk.representation.PaiNN(n_atom_basis = args.features,
                                    n_interactions = args.interactions,
                                    radial_basis = radial_basis,
                                    cutoff_fn = spk.nn.CosineCutoff(args.cutoff))
if args.schnet:
    repr = spk.representation.SchNet(n_atom_basis=args.features,
                                     n_interactions=args.interactions,
                                     radial_basis=radial_basis,
                                     cutoff_fn=spk.nn.CosineCutoff(args.cutoff))

# Energy module 
pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis,output_key="energy", n_layers=args.layer)
pred_forces =  spk.atomistic.Forces(energy_key="energy", force_key="forces")#(energy_key="energy",force_key="forces",n_states=n_ener)
nnpot = spk.model.NeuralNetworkPotential(
    representation=repr,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy,pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy",add_mean=True,add_atomrefs=False,is_extensive=False)
    ]
)

output_energy = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=args.rho,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)
output_forces = spk.task.ModelOutput(
    name="forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1-args.rho,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": args.lr},
    scheduler_cls=spk.train.ReduceLROnPlateau,
    scheduler_args={"patience": args.lr_patience,
                    "min_lr": args.lr_min,
                    "factor": args.lr_decay
    },
    scheduler_monitor="val_loss"
)

logger = pl.loggers.TensorBoardLogger(save_dir=args.rootdir)
callbacks = [
    spk.train.ModelCheckpoint(
        inference_path=os.path.join(args.rootdir, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

if args.gpu:
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.rootdir,
        max_epochs=max_epochs, # for testing, we restrict the number of epochs
        devices=args.gpu,
        accelerator="gpu"
    )
else:
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.rootdir,
        max_epochs=max_epochs, # for testing, we restrict the number of epochs
    )

trainer.fit(task, datamodule=dataset)

