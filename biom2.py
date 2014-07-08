import sys
import numpy as np
import subprocess  

in_file_txt = sys.argv[1]
out_file_b = sys.argv[1] + ".biom"

name = sys.argv[1].split(".txt")[0]

procentage = int(sys.argv[2])

s = np.loadtxt(in_file_txt, dtype="string")
ncol = s.shape[1] - 2

txt2biom = ["biom", "convert", "-i", in_file_txt, "-o", out_file_b, "--table-type", "otu table"]

process1 = subprocess.Popen(txt2biom)

process1.wait()

out_file_filter = name +"_filter.biom"

filter_otus = ["filter_otus_from_otu_table.py", "-i", out_file_b, "-o", out_file_filter, "-s", str(ncol*procentage/100)]

process2 = subprocess.Popen(filter_otus)

process2.wait()

out_file_filter_txt = name + "_filter.txt"

biom2txt = ["biom", "convert", "-i", out_file_filter, "-o", out_file_filter_txt, "-b", "--header-key=Taxon"]

process3 = subprocess.Popen(biom2txt)



