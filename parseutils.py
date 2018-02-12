# Author: Abhishek Munagekar
# Parsing Utilities - Extension for Argparse Types
# Language Python3

import argparse


# Integer Greater than zero
def intg0(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an Integer" % (value,))
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "%r not a positive integer" % (value,))
    return value


# Float in [0,1]
def floatnorm(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a Float" % (value,))
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError(
            "%r not a positive Float <=1" % (value,))
    return value


# Float >=0
def pfloat(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentError("%r not a Float" % (value,))
    if value <= 0:
        raise argparse.ArgumentError("%r not a positive Float" % (value,))
    return value
