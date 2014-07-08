import subprocess 
import sys
import numpy as np

in_file = sys.argv[1]

name = sys.argv[1].split(".biom")[0]

percentage = int(sys.argv[2])

biom2txt = ["biom", "convert", "-i", in_file, "-o", name + ".txt", "-b", "--header-key=taxonomy"]

process = subprocess.Popen(biom2txt)
process.wait()

s = np.loadtxt(name + ".txt", dtype="string", delimiter='\t')
ncol = s.shape[1] - 2

out_file = name + "_filter_taxa.biom"

filter_taxa_from_otu_table = ["filter_taxa_from_otu_table.py", "-i", in_file, "-o", out_file, "-n", "Unassigned"]

process1 = subprocess.Popen(filter_taxa_from_otu_table)
process1.wait()

in_file = out_file
out_file = name +"_f_tax_otu.biom"

filter_otus_from_otu_table = ["filter_otus_from_otu_table.py", "-i", in_file, "-o", out_file, "-s", str(ncol*percentage/100)]

process2 = subprocess.Popen(filter_otus_from_otu_table)
process2.wait()

in_file = out_file

summarize_taxa = ["summarize_taxa.py", "-i", in_file, "-o", "./summarized_taxa"]

process3 = subprocess.Popen(summarize_taxa)
process3.wait()

process4 = subprocess.Popen(summarize_taxa + ["-L 7"])