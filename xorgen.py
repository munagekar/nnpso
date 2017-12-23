'''
Xor Dataset Generator : Python 3
Author: Abhishek Munagekar
'''
import random

#Parameters




xor_num = 5
nos_entry = 10000
_seed = 3
dumpfilename='dump.txt'


string = ['0','1']
random.seed(_seed)

dump = open(dumpfilename,'w')
for i in range (nos_entry):
	_input = [None]* xor_num
	_output = 0
	for j in range(xor_num):
		_input[j]=random.randint(0,1)
		dump.write(string[_input[j]]+"\t")
		_output ^=_input[j]
	dump.write(string[_output]+"\n")
dump.close()








