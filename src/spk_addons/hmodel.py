import schnetpack as spk
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
from schnetpack import properties
from schnetpack.atomistic import Atomwise
import schnetpack.nn as snn
from .data import *
from typing import Callable, Dict, Sequence, Tuple, List, Optional, Union

class Atomwise_adapted(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        super(Atomwise_adapted, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                for i in range(len(y[0])):
                    y[:,i] = y[:,i] / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs

class TwoSpeciesSystem_Diab(nn.Module):


    # Predicts diabatic hamiltonian for a two-species system, e.g., hybrid inorganic-organic interfaces, molecules at metal surfaces, two-molecule systems, ...
    # uses adiabatic (orbital) energies of separate species and puts them together into a H matrix. This H matrix is diagonalized and the eigenvalues are mapped to the adiabatic (orbital) energies of the system that contains both species. 

    # uses aggregation_mode avg as default

    def __init__(
        self,
        calc_forces:  bool = False,
        calc_stress:  bool = False,
        species1_key: str  = "energy_1", #eigenvalues_metal
        species2_key: str  = "energy_2", #eigenvalues_ad
        forces1_key:  str  = "forces_1",
        forces2_key:  str  = "forces_2",
        n_states_1:   int  = 1,
        n_states_2:   int  = 1,
        couplings:    str  = "couplings",
        output_key:   str  = "energy",
    ):
        """
        Args:
            if calc_forces calc_stress = True: compute forces or stress (PBC)
            energy keys: name of energy inputs for the different species
            if forces requires provide force keys
            n_states_1 number of states of species 1
            n_states_2 number of states of species 2
        """
        super(TwoSpeciesSystem_Diab, self).__init__()
        self.calc_forces  = calc_forces
        self.calc_stress  = calc_stress
        self.species1_key = species1_key
        self.species2_key = species2_key
        self.forces1_key  = forces1_key
        self.forces2_key  = forces2_key
        self.n_states_1   = n_states_1
        self.n_states_2   = n_states_2
        self.couplings    = couplings
        self.output_key   = output_key
        self.n_states_t   = n_states_1 + n_states_2
        self.model_outputs = [output_key]
        # derivatives not implemented yet  
        self.required_derivatives = []
        if self.calc_forces == True:
            	self.required_derivatives.append(properties.R)
        if self.calc_stress == True:
                self.required_derivatives.append(properties.strain)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # get E1 and E2
        # the energies of separate systems are obtained domain-specifically
        # E1 is obtained from averaging as many outputs as there are in system1_key, but only using atomistic features of species 1
        # E2 is obtained from averaging as many outputs as there are in system2_key, but only using atomistic features of species 2
        # C is obtained from both, outputs that belong to species 1 and 2.
	# all will be put together and diagonalized. 
        # eigenvalues will be mapped to the energies obtained from both species put together.

        E1 = inputs[self.species1_key]
        E2 = inputs[self.species2_key]
        C  = inputs[self.couplings]
        # put together hamiltonian and get eigenvalues
        inputs[self.output_key] = self.get_hamiltonian(E1,E2,C)
        return inputs

    def get_hamiltonian(self,E1,E2,C):
        #print(E1,E2,C)
        H = torch.diag_embed( torch.cat((E1,E2),dim=1), dim1=-2, dim2=-1)
        # only fill upper triangular, lower is ignored anyways during diagonalization
        #print(H.shape,C.shape,)
        it=-1
        for istate in range(self.n_states_t):
            for jstate in range(istate+1, self.n_states_t):
                it+=1
                H[:,istate,jstate] = C[:,it]
        #print(H,it)
        eigenvalues, eigenvectors = torch.linalg.eigh(H,UPLO="U")
        #torch.zeros(len(E1)+len(E2),len(E1)+len(E2))
        
        return eigenvalues

        
