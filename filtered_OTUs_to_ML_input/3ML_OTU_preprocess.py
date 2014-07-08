import numpy as np
import sys
import argparse
import os
import fnmatch
import time

class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


parser = myArgumentParser(description='Preprocess OTU files for ML algorithm by MC and AZ',
        fromfile_prefix_chars='@')
 
#parser.add_argument('DATAFILE', type=str, help='Training datafile')       
parser.add_argument('OTUs_DIR', type=str, help='Directory with the otu_table_L?.txt files')
parser.add_argument('IN_Y', type=str, help='Y labels of samples (healthy/ill)')
parser.add_argument('OUTDIR', type=str, help='Output directory')
parser.add_argument('--id_name', type=str, default='merged', help='Name of merged IDs (default: "merged")')
parser.add_argument('--exp_merged_OTUs', action="store_true", help='Use this option to get the merged OTU table')


if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
OTUs_DIR=args.OTUs_DIR
in_y=args.IN_Y
outdir=args.OUTDIR
idname=args.id_name
expMergedOTU=args.exp_merged_OTUs





if OTUs_DIR[-1]!='/':
	OTUs_DIR+='/'

def get(datafile):
	text=np.loadtxt(datafile,delimiter='\t',dtype=str,skiprows=1).tolist()
	return text
		
def get2(datafile):
	f=open(datafile,'r')
	text=[]
	for line in f:
		text.append(line)
	text.pop(0)
	return text
		
	



otusfiles=[]
for file in os.listdir(OTUs_DIR):
	if fnmatch.fnmatch(file,'otu_table_L?.txt'):
		otusfiles.append(file)
	
otusfiles=np.sort(otusfiles)	
#print otusfiles


premerged=[]  #np.empty([0,0])

#out=open(OTUs_DIR+"output.txt",'w')
#out=tempfile.NamedTemporaryFile(mode='w'+'b'+'r')

outname="/tmp/premergefile"+str(time.time())
out=open(outname,'w')

#print outname

#print open(OTUs_DIR+otusfiles[0],'r').readline()

out.write(open(OTUs_DIR+otusfiles[0],'r').readline())

for file in otusfiles:
	f=open(OTUs_DIR+file)
	f.readline()
	for line in f:
		out.write(line)

out.close()

data=np.loadtxt(outname,dtype=str,delimiter='\t')

os.unlink(outname)


merged= np.append(data,data[:,0].reshape(len(data),1),axis=1)

for i in range(1,len(merged)):
	merged[i,0]=idname+np.str(i-1)
merged[0,0]='#OTU ID'
	
#print merged

#print expMergedOTU

if not(os.path.exists(outdir)):
	os.mkdir(outdir)

if expMergedOTU==True:
	np.savetxt(outdir+"/merged_otus.txt",merged,fmt="%s",delimiter='\t',header='Constructed from biom file')


#!/usr/bin/python

import os

#infile="merged_otus.txt"
#in_y="Y_without_met.txt"
#outdir="2otu_data/"

if outdir[-1]=='/':
	outdir=outdir[:-1]


#data=np.loadtxt(infile,dtype=str,comments="^",delimiter='\t',skiprows=1)

data=merged


X=np.sort(np.transpose(data[:,1:-1],(1,0)),axis=0)[:,1:].astype(float)  #sorts the data by ID after having flipped the rows and columns and removed names of bacteria, then removes first column (IDs) and converts the numbers into floats

pre_Y=np.loadtxt(in_y,delimiter='\t',dtype=str)

if(len(pre_Y[0])>1):      
	pre_Y.sort(axis=0)
	Y=pre_Y[:,-1].astype(int)
else:
	Y=pre_Y.astype(int)

#print pre_Y



#Y=np.loadtxt(in_y,delimiter='\t',usecols=(1,),dtype=int)

#labels=data[1:,(0,-1)]
labels=data[1:,0]
names=data[1:,(0,-1)]
sampleIDs=np.sort(np.transpose(data[:,1:-1],(1,0)),axis=0)[:,0]

'''
print X
print X.shape
print labels
print names
print Y
print sampleIDs'''


np.savetxt(outdir+"/X.txt",X[:-3,],delimiter='\t',newline='\n',fmt="%.8f");  #-3 !!!!
#X.tofile("otu_data/X.txt",sep='\t',format="%.1f");
np.savetxt(outdir+"/Y.txt",Y,fmt="%d");
np.savetxt(outdir+"/labels.txt",labels,fmt="%s");
np.savetxt(outdir+"/names.txt",names,delimiter='\t',fmt="%s");
np.savetxt(outdir+"/sampleIDs.txt",sampleIDs,fmt="%s");



