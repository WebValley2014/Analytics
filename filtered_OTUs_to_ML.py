import numpy as np
import sys
import argparse


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
parser.add_argument('INFILE', type=str, help='Filtered and merged OTU file')
parser.add_argument('IN_Y', type=str, help='Y labels of samples (healthy/ill)')
parser.add_argument('OUTDIR', type=str, help='Output directory')



if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
infile=args.INFILE
in_y=args.IN_Y
outdir=args.OUTDIR



import os

#infile="merged_otus.txt"
#in_y="Y_without_met.txt"
#outdir="2otu_data/"

if outdir[-1]=='/':
	outdir=outdir[:-1]


data=np.loadtxt(infile,dtype=str,comments="^",delimiter='\t')



X=np.sort(np.transpose(data[:,1:-1],(1,0)),axis=0)[:,1:].astype(float)  #sorts the data by ID after having flipped the rows and columns and removed names of bacteria, then removes first column (IDs) and converts the numbers into floats
Y=np.loadtxt(in_y,delimiter='\t',usecols=(1,),dtype=int)

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

if not(os.path.exists(outdir)):
	os.mkdir(outdir)
np.savetxt(outdir+"/X.txt",X[:,],delimiter='\t',newline='\n',fmt="%.8f");  #-3 !!!!
#X.tofile("otu_data/X.txt",sep='\t',format="%.1f");
np.savetxt(outdir+"/Y.txt",Y,fmt="%d");
np.savetxt(outdir+"/labels.txt",labels,fmt="%s");
np.savetxt(outdir+"/names.txt",names,delimiter='\t',fmt="%s");
np.savetxt(outdir+"/sampleIDs.txt",sampleIDs,fmt="%s");



