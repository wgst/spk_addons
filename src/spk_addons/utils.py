import os
import numpy as np
import sys
import argparse
import logging
from os import getcwd


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
