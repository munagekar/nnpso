'''
This file contains the new network definition
Author: Abhishek Munagekar
Language: Python 3
'''
#The number of values for xor


import itertools
from functools import reduce
import operator
N_IN = 5

#Generate the batch dataset here

#A list of lists having N_IN elements all either 0 or 1
xor_in = [list(i) for i in itertools.product([0, 1], repeat=N_IN)]
#A list having 2^N lists each having xor of each input list in the list of lists
xor_out = list(map(lambda x: [(reduce(operator.xor,x))],xor_in))

