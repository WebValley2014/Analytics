import numpy as np
import os
import re


def match(arr,col,regex):
	ok=[]
	for i in range(len(arr)):
		if re.match(regex,arr[i,col])!=None:
			ok.append(i)
	#print ok
	return arr[ok]
	
def mysort(arr):
	lst=arr.tolist()
	lst.sort()
	return np.array(lst)
	

def export(set1,set2,outname):
	y_wo=[]
	#print outname
	'''for elem in lst:
		y_wo.append(elem)'''
	
	y_wo=np.append(set1,set2,axis=0)
	#print y_wo
	y_wo=mysort(y_wo)
	#print y_wo
	
	sampleIDs=y_wo[:,0]
	#print "sampleids",sampleIDs
	#print "len",len(sampleIDs)
	xrows=[0]
	#print data[0].tolist()
	for name in sampleIDs:
		data[0].tolist().index(name)
		xrows.append(data[0].tolist().index(name))
	xrows.append(-1)
	outOTU=data[:,xrows]
	#print outOTU
	#print len(outOTU[0])
	np.savetxt(outdir+"/"+outname+"_OTUtable.txt",outOTU,delimiter='\t',fmt="%s")
	np.savetxt(outdir+"/"+outname+"_Y_labels.txt",y_wo,delimiter='\t',fmt="%s")
		



infile="4classif/otu_table.txt"
in_y="4classif/Y_without_met.txt"
outdir="4classif/comparisons/"

if outdir[-1]=='/':
	outdir=outdir[:-1]


data=np.loadtxt(infile,dtype=str,comments="^",delimiter='\t')  #skiprows=1
#print data.shape
#print data[:,0]
#print np.transpose(data[:,1:-1],(1,0))#.astype(float)

data[0,-1]='taxonomy'

X=np.sort(np.transpose(data[:,1:-1],(1,0)),axis=0)[:,1:].astype(float)  #sorts the data by ID after having flipped the rows and columns and removed names of bacteria, then removes first column (IDs) and converts the numbers into floats

#print X.shape
#print X

pre_Y=np.loadtxt(in_y,delimiter='\t',dtype=str)

'''
if(len(pre_Y[0])>1):      
	pre_Y.sort(axis=0)
	Y=pre_Y[:,-1].astype(int)
else:
	Y=pre_Y.astype(int)'''

np.sort(pre_Y)
pre_Y=mysort(pre_Y)

#print "pre_y",pre_Y


FEC_H=match(pre_Y,0,"neg\.[A-Za-z0-9]+$")
B_IBD=match(pre_Y,0,"^[A-Za-z0-9]+\.[A-Za-z0-9]+\.B$")
B_H=match(pre_Y,0,"^[A-Za-z0-9]+\.[A-Za-z0-9]+\.A$")
FEC_IBD=match(pre_Y,0,"[A-Z0-9]+\.[A-Za-z0-9]+$")

labels=data[1:,0]
names=data[1:,(0,-1)]
#print names

'''
print FEC_H
print B_IBD
print B_H
print FEC_IBD
'''

if not(os.path.exists(outdir)):
	os.mkdir(outdir)

#print data

export(FEC_H,FEC_IBD,"FEC_H_IBD")
export(B_H,B_IBD,"B_H_IBD")
export(FEC_H,B_H,"FEC_H_B_H")
export(FEC_IBD,B_IBD,"FEC_B_IBD")




