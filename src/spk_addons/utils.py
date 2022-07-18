import os
import numpy as np
import sys
import argparse
import logging
from os import getcwd
import yaml

import schnetpack as spk

def get_representation(args):

    radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_gaussian, cutoff = args.cutoff)

    if args.painn:
        representation = spk.representation.PaiNN(n_atom_basis = args.features,
                                    n_interactions = args.interactions,
                                    radial_basis = radial_basis,
                                    cutoff_fn = spk.nn.CosineCutoff(args.cutoff))
    if args.schnet:
        representation = spk.representation.SchNet(n_atom_basis = args.features,
                                    n_interactions = args.interactions,
                                    radial_basis = radial_basis,
                                    cutoff_fn = spk.nn.CosineCutoff(args.cutoff))
    return representation

def read_tradeoffs(filename):
    #Properties = ['system1', 'system2', 't_system1', 't_system2']
    with open(filename, 'r') as tf:
        propdict = yaml.safe_load(tf)

    return propdict

def read_param(filename):

    params = {}

    file = open(filename,"r").readlines()
    for line in file:
        key = str(line.split()[0])
        value = line.split()[1]
        params[key]=value
    return params

def log_arguments(args):
    """
    This function creates a logfile with training options from the passed arguments.
        args: argparse object
    """

    log = logging.getLogger(__name__)
    path = args['rootdir']
    log.info(f"Writing arguments to {path}/args.json")

    try:
        logfile = open(f"{path}/args.json","w")
    except OSError:
        print(f"Cannot open {path}")

    log = ""
    for key in args.keys():
        value = args[key]
        log+=f"{key} {value}"
        log+="\n"
    logfile.write(log)
    logfile.close()
