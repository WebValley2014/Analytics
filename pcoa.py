#!/usr/bin/python

import numpy as np
import os, sys

def makepcoa(beta_div_pcoa_file):
	data = np.genfromtxt(beta_div_pcoa_file, skip_header = 1, skip_footer=2, dtype=str)
	for i in range(len(data)):
		data[i][-1]+=" "
	
	np.savetxt('pcoa_data.txt',data,delimiter=' ',fmt='%s')

if __name__ == '__main__':
	makepcoa(sys.argv[1])