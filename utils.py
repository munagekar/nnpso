# Author : Abhishek Munagekar
# Misc Utils for the Project
# Language: Python 3

from datetime import datetime
import psutil
import os


# Data Storage Constants

DataUnits = {'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024 ** 3}


# Return the current time in a easy to print string format


def curtime():
    return str(datetime.now())


# Returns the current memory usage by the python process
def str_memusage(datatype):
    process = psutil.Process(os.getpid())
    assert datatype in DataUnits, 'Datatype must be in [B,M,G]'
    return str(process.memory_info().rss / DataUnits[datatype]) + datatype
