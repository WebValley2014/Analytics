## This code is written by Davide Albanese <albanese@fbk.eu> and Marco Chierici <chierici@fbk.eu>
# CHANGELOG
# 2014-03-13: [AZ]  moved "C" parameter from positional to optional (-c)
# 2014-03-13: [AZ]  added check on parameter "C": if not available, tuning step will be performed (svmlin_t)
# 2014-03-13: [AZ]  modified input requirements so to start from full TR/TS datasets and extract top N features internally (by using extract_feats function from extract_topfeats_nb.py)

from __future__ import division
import numpy as np
import os.path
import mlpy
from input_output import load_data
import performance as perf
from scaling import *
import argparse
import sys
from extract_topfeats_nb2 import extract_feats

parser = argparse.ArgumentParser(description = 'Run a validation experiment using LibLinear (with a user-provided C, optionally). Currently only with minmax and SVM')
parser.add_argument('TRFILE', type = str, help = 'Training datafile')
parser.add_argument('TSFILE', type = str, help = 'Validation datafile')
parser.add_argument('TR_SAMPLE_IDs', type = str, help = 'Training set sample ID datafile')
parser.add_argument('TS_SAMPLE_IDs', type = str, help = 'Test set sample ID datafile')
parser.add_argument('FEATURES_IDs', type = str, help = 'Feature ID datafile. The IDs must match with those in the ranking file')
parser.add_argument('LABELSFILE', type = str, help = 'Training labels')
#parser.add_argument('SCALING', type = str, choices = ['norm_l2', 'std', 'minmax'], default = 'norm_l2', help = 'Scaling method')
#parser.add_argument('SVM_TYPE', type = str, choices = ['l2r_l2loss_svc', 'l2r_l2loss_svc_dual', 'l2r_l1loss_svc_dual', 'l2r_lr_dual', 'l1r_l2loss_svc'], help = 'SVM type')
parser.add_argument('RANK', type = str, help = 'Features list ranked by training experiment')
#parser.add_argument('NFEATS', type = np.int, help = 'Number of ranked features to consider for validation')
parser.add_argument('ML_FEATURES', type = str, help= 'Machine learning algorithm feature datafile (containing scaling method, SVM type, number of features)')
parser.add_argument('CONFIDENCE_PERCENT', type = float, default=60, help = 'Threshold percent probability for diagnosis >50 percent')
parser.add_argument('OUTDIR', type = str, help = 'Output directory')
parser.add_argument('-c', type=np.float, default=None, help='SVM C parameter, if available')
parser.add_argument('--tslab', type=str, default=None, help='Validation labels, if available')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
TRFILE = vars(args)['TRFILE']
TSFILE = vars(args)['TSFILE']
FEAT_IDS = vars(args)['FEATURES_IDs']
TR_SAMPLE_IDS = vars(args)['TR_SAMPLE_IDs']
TS_SAMPLE_IDS = vars(args)['TS_SAMPLE_IDs']
LABELSFILE = vars(args)['LABELSFILE']
#SCALING = vars(args)['SCALING']
#SVM_TYPE = vars(args)['SVM_TYPE']
C = vars(args)['c']
RANK = vars(args)['RANK']
#NFEATS = vars(args)['NFEATS']
OUTDIR = vars(args)['OUTDIR']
CONF_PERC = vars(args)['CONFIDENCE_PERCENT']
TSLABELSFILE = vars(args)['tslab']
ML_FEATURES=vars(args)['ML_FEATURES']

mlFeatures=np.loadtxt(ML_FEATURES,delimiter='\t',dtype=str)
SCALING=mlFeatures[0]
SVM_TYPE=mlFeatures[1]
NFEATS=np.int(mlFeatures[2])

if(OUTDIR[-1]!='/'):
	OUTDIR+='/'

if not(os.path.exists(OUTDIR)):
	os.mkdir(OUTDIR)


def compute_weights(y):
    classes = np.unique(y)
    weight = {}
    for c in classes:
        weight[c] = y.shape[0] / np.sum(y==c)
    return weight

BASEFILE = os.path.splitext(TRFILE)[0]
BASEFILENAME = os.path.split(BASEFILE)[1]
OUTFILE = BASEFILE

"""
# extract the top-ranked NFEATS features from Validation set
TS_TOPFEATS = OUTFILE + 'ts_topfeats.txt' 
extract_feats(TSFILE, SAMPLE_IDS, FEAT_IDS, RANK, NFEATS, TS_TOPFEATS)
# extract the top-ranked NFEATS features from Training set
TR_TOPFEATS = OUTFILE + 'tr_topfeats.txt' 
extract_feats(TRFILE, SAMPLE_IDS, FEAT_IDS, RANK, NFEATS, TR_TOPFEATS)

# load data
sample_names_tr, var_names_tr, x_tr = load_data(TR_TOPFEATS)
print "samplenames",sample_names_tr
print "varnames",var_names_tr
print "Xtr",x_tr
sample_names_ts, var_names_ts, x_ts = load_data(TS_TOPFEATS)"""

y_tr = np.loadtxt(LABELSFILE, dtype=np.int, delimiter='\t')

TS_TOPFEATS = OUTFILE + 'ts_topfeats.txt' 
TR_TOPFEATS = OUTFILE + 'tr_topfeats.txt' 

sample_names_tr, var_names_tr, x_tr =extract_feats(TRFILE, TR_SAMPLE_IDS, FEAT_IDS, RANK, NFEATS, TR_TOPFEATS)
sample_names_ts, var_names_ts, x_ts =extract_feats(TSFILE, TS_SAMPLE_IDS, FEAT_IDS, RANK, NFEATS, TS_TOPFEATS)

print "samplenames",sample_names_tr
print "varnames",var_names_tr
print "Xtr",x_tr



# load the TS labels if available
if TSLABELSFILE is not None:
    y_ts = np.loadtxt(TSLABELSFILE, dtype=np.int, delimiter='\t')

if SVM_TYPE!='randomForest':
	# tuning step if parameter C is not available
	if C is None: 
	    from svmlin_tuning import svmlin_t
	    TUN_CV_K = 10
	    TUN_CV_P = 50
	    TUN_SVM_C = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 
			 0.01, 0.1, 1, 10, 100, 1000, 10000] 
	    C, mcc, err, mcc_tr, err_tr = svmlin_t(x_tr, y_tr, svm_type=SVM_TYPE, scaling=SCALING, list_C=TUN_SVM_C, cv_k=TUN_CV_K, cv_p=TUN_CV_P)
	else:
	    print "Tuning with parameter C from user..."

# centering and normalization
if SCALING == 'norm_l2':
    x_tr, m_tr, r_tr = norm_l2(x_tr) 
    x_ts, _, _ = norm_l2(x_ts, m_tr, r_tr) 
elif SCALING == 'std':
    x_tr, m_tr, r_tr = standardize(x_tr) 
    x_ts, _, _ = standardize(x_ts, m_tr, r_tr)            
elif SCALING == 'minmax':
    x_tr, m_tr, r_tr = minmax_scaling(x_tr) 
    x_ts, _, _ = minmax_scaling(x_ts, m_tr, r_tr)

# prediction

if SVM_TYPE!='randomForest':
	svm_w = compute_weights(y_tr)
	svm = mlpy.LibLinear(solver_type=SVM_TYPE, C=C, weight=svm_w)
	svm.learn(x_tr, y_tr)
	p_tr = svm.pred(x_tr)
	p_ts = svm.pred(x_ts)
	
	prob_tr = svm.pred_probability(x_tr)
	prob_ts = svm.pred_probability(x_ts)
else:
	from sklearn.ensemble import RandomForestClassifier
	forest=RandomForestClassifier()
	forest.fit(x_tr,y_tr)
	p_tr=forest.predict(x_tr)
	p_ts=forest.predict(x_ts)
	prob_tr = forest.predict_proba(x_tr)
	prob_ts = forest.predict_proba(x_ts)
	

print "MCC on train: %.3f" % (perf.KCCC_discrete(y_tr, p_tr))
if TSLABELSFILE is not None:
    print "MCC on validation: %.3f" % (perf.KCCC_discrete(y_ts, p_ts))



# write output files
fout = open(OUTDIR+BASEFILENAME + "_TEST_pred_tr.txt", "w")
for i in range(len(sample_names_tr)):
    fout.write("%s\t%i\n" % (sample_names_tr[i], p_tr[i]))
fout.close()

fout = open(OUTDIR+BASEFILENAME + "_TEST_pred_ts.txt", "w")
for i in range(len(sample_names_ts)):
    fout.write("%s\t%i\n" % (sample_names_ts[i], p_ts[i]))
fout.close()

np.savetxt(OUTDIR+BASEFILENAME + "_TEST_signature.txt",
           np.array(var_names_tr).reshape(-1,1),
           fmt='%s', delimiter='\t')

fout = open(OUTDIR+BASEFILENAME + "_TEST_prob_tr.txt", "w")
fout.write("SAMPLE\tCLASS 1\tCLASS 0\n")
for i in range(len(sample_names_tr)):
    fout.write("%s\t%f\t%f\n" % (sample_names_tr[i], prob_tr[i,0], prob_tr[i,1]))
fout.close()

fout = open(OUTDIR+BASEFILENAME + "_TEST_prob_ts.txt", "w")
fout.write("SAMPLE\tCLASS 1\tCLASS 0\n")
for i in range(len(sample_names_ts)):
    fout.write("%s\t%f\t%f\n" % (sample_names_ts[i], prob_ts[i,0], prob_ts[i,1]))
fout.close()


statuses=[]
for i in range(len(sample_names_ts)):
	if prob_ts[i,0]*100<(100-CONF_PERC):
		statuses.append(1)
	elif prob_ts[i,0]*100>CONF_PERC:
		statuses.append(-1)
	else: #(prob_ts[i,0]*100>(100-CONF_PERC)) and (prob_ts[i,0]*100<CONF_PERC)):
		statuses.append(0)
		
dict={0:"There is not enough evidence to determine whether the patient is healthy or ill in the confidence interval", 1:"The patient appears to be healthy in the confidence interval", -1: "The patient appears to be ill in the confidence interval"}		
		
	
fout = open(OUTDIR+BASEFILENAME + "_DIAGNOSIS_ts.txt", "w")
fout.write("SAMPLE\tPRED_LABEL\tP(ILL)\tP(HEALTHY)\tCONF_PERC\tDIAGNOSIS\n")
for i in range(len(sample_names_ts)):
    fout.write("%s\t%d\t%f\t%f\t%f\t%s (%.2f percent)\n" % (sample_names_ts[i], p_ts[i], prob_ts[i,0], prob_ts[i,1], CONF_PERC, dict[statuses[i]], max(prob_ts[i,0],prob_ts[i,1])*100))
fout.close()

