#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os, sys

def process_otu_table(): #<--set parameters
	n_groups = 10 #how many ranked features to keep
	
	#output from machine learning
	featureNames = np.loadtxt('X_l2r_l2loss_svc_dual_SVM_std_featurelist.txt', dtype=str, skiprows=1, usecols=(0,1))
	ranked = featureNames[:,1] #names of feautres
	
	wholetable = np.loadtxt('otu_table.txt',dtype=str, delimiter='\t', comments='^')
	namecolumn = wholetable[:,0]
	
	processedtable = np.empty((n_groups+2,len(wholetable[1,:])),dtype='S512') #just keep abundances of ranked features
	processedtable[0,:] = wholetable[0,:]
	
	#write labels (0 diseased, 1 healthy) into second row of processed otu  table
	# 1 column file with 1's and 0's
	labels = np.loadtxt('Y.txt',dtype=int, delimiter='\n')
	processedtable[1,0] = "Label"
	processedtable[1,1:]= labels
	
	#start writing in data from ranked features from row 3
	current_row = 2
		for f in ranked[:n_groups]:
		for i in range(1, len(namecolumn)):
			if f in namecolumn[i] and current_row<=n_groups:
				processedtable[current_row,:] = wholetable[i,:]
				current_row+=1

	processedtable.transpose()

	#write output to file
	np.savetxt('processed_otu_table.txt',processedtable,delimiter='\t',fmt='%s')

if __name__ == '__main__':
	#parameters: n_groups (how many features), feature list output from  machine learning, otu table, 1 column labels file (0 and 1)
	processed_otu_table(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])