import os
filelist = os.listdir(os.getcwd())

filename = [f for f in filelist if 'apping' in f][0]

import numpy as np
mapping = np.loadtxt(filename, dtype="string")

import csv
outw = open("oligos_File1111.txt", "w")
writer = csv.writer(outw, delimiter='\t', lineterminator='\n')

for a in range(mapping.shape[0]):
	if a == 0:
		writer.writerow( ["forward", mapping[2,2] ] )
		writer.writerow( ["barcode", mapping[a,1], mapping[a,0]] )
	else:
		writer.writerow( ["barcode", mapping[a,1], mapping[a,0]] )

outw.close()




