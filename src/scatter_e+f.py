import matplotlib.pyplot as plt
import numpy as np
import os
import sys


size=22
params = {'legend.fontsize': 'large',
          'figure.figsize': (7,5),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size,
          'axes.titlepad': 10,
         'axes.linewidth':1.5}
plt.rcParams.update(params)

path = sys.argv[1]
reference = np.load(path+"/reference.npz",allow_pickle=True)["arr_0"][()]
predictions = np.load(path+"/predictions.npz",allow_pickle=True)["arr_0"][()]

energy_qc = []# reference["energy"]
energy_nn = []#predictions
forces_qc = []
forces_nn = []
for ibatch in range(len(reference["energy"])):
    for iener in range(len(reference["energy"][ibatch])):
        energy_qc.append(reference["energy"][ibatch][iener])#.detach().numpy())
        energy_nn.append(predictions["energy"][ibatch][iener].detach().numpy())
    for iforce in range(len(reference["forces"][ibatch])):
        for xyz in range(3):
            forces_qc.append(float(reference["forces"][ibatch][iforce][xyz]))
            forces_nn.append(float(predictions["forces"][ibatch][iforce][xyz].detach().numpy()))
forces_qc = np.array(forces_qc)
forces_nn = np.array(forces_nn)
energy_nn = np.array(energy_nn)
energy_qc = np.array(energy_qc)
MAE = np.mean(np.abs(energy_nn-energy_qc))
M = np.abs ( energy_nn-energy_qc)
plt.scatter(energy_nn,energy_qc,label="MAE:%f"%MAE)
plt.xlabel("Energy NN")
plt.ylabel("Energy QC")
plt.legend(fontsize="x-large")
plt.savefig(path+"/Energy.png",dpi=300,transparent=True,bbox_inches="tight")

MAEf =np.mean(np.abs(forces_nn-forces_qc))
plt.scatter(forces_nn,forces_qc,label="MAE:%f"%MAEf)
plt.xlabel("Forces NN")
plt.ylabel("Forces QC")
plt.legend(fontsize="x-large")
plt.savefig(path+"/Forces.png",dpi=300,transparent=True,bbox_inches="tight")

os.system('echo "MAE Energy %12.9f" > %s/MAE.txt'%(MAE,path))
os.system('echo "MAE Forces %12.9f" >> %s/MAE.txt'%(MAEf,path))

