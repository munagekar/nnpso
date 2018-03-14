# Author : Abhishek Munagekar
# Misc Utils for the Project
# Language: Python 3

from datetime import datetime
import psutil
import operator
import os
from functools import reduce

# Data Storage Constants
DataUnits = {'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024 ** 3}


# Print the current time along with a string
def msgtime(prefix=''):
    print(prefix + str(datetime.now()))


# Returns the current memory usage by the python process
def str_memusage(datatype):
    process = psutil.Process(os.getpid())
    assert datatype in DataUnits, 'Datatype must be in [B,M,G]'
    return str(process.memory_info().rss / DataUnits[datatype]) + datatype


# Progress Bar Printing from
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_prog_bar(iteration, total, prefix='', suffix='',
                   decimals=2, length=90, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    strdec = str(decimals)
    percent = ("{0:." + strdec + "f}").format(100 *
                                              (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# Fully Connected Network Information Printer
# Prints the stats given the network definition
# Net def is a list of numbers
def fcn_stats(net_def):
    num_l = len(net_def)
    num_hl = num_l - 2
    num_weights = reduce(operator.mul, net_def)
    num_biases = reduce(operator.mul, net_def[1:])
    tot_dim = num_weights + num_biases
    print('#*******NET STATS*******#')
    print('Layers\t\t\t:', num_l)
    print('Hidden\t\t\t:', num_hl)
    print('Weight Dims\t\t:', num_weights)
    print('Bias Dims\t\t:', num_biases)
    print('Total Dims\t\t:', tot_dim)
