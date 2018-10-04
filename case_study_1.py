# -*- coding: utf-8 -*-
"""
CASE STUDY 1 - DNA TRANSLATION

"""
# first step download the nucleotide sequence from
# NCBI using the name code NM_207618.2 in FASTA
# e.g. copy-paste the DNA sequence into the editor
# then download the CDS amino acid sequence
# 2 files are now present, nucleotides.txt and aminoacids.txt
# we need to remove the special characters such as \n
# first we load the file and read the content

def read_seq(input_file):
	"""Reads and returns the input sequence with special
	characters removed."""
	with open(input_file, "r") as file: # read only
		seq = file.read()

	seq = seq.replace('\n', '') # replace \n by nothing
	seq = seq.replace('\r', '') # hidden character 'carriage return'

	return seq

###########
### Getting the same information using biopython
###########

from Bio import Entrez
from Bio import SeqIO
email = 'sbwiecko@free.fr'
term = 'NM_207618.2' #accession/version

h_search = Entrez.esearch(
		db='nucleotide', email=email, term=term)
record = Entrez.read(h_search)
h_search.close()

handle_nt = Entrez.efetch(
		db='nucleotide', email=email, 
		id=record['IdList'][0], rettype='fasta')
results = Entrez.read(Entrez.elink(
		dbfrom='nucleotide', linkname='nucleotide_protein',
		email=email, id=record['IdList'][0]))
handle_aa = Entrez.efetch(
		db='protein', email=email, 
		id=results[0]['LinkSetDb'][0]['Link'][0]['Id'], 
		rettype='fasta')

nucleotide_seq = str(SeqIO.read(handle_nt, format='fasta').seq)
aminoacid_seq  = str(SeqIO.read(handle_aa, format='fasta').seq)

handle_nt.close(); handle_aa.close()

###########
def translate(seq):
	"""Translate a string containing a nucleotide sequence 
	into a string containing the corresponding sequence of 
	amino acids . Nucleotides are translated in triplets using 
	the table dictionary; each amino acid 4 is encoded with 
	a string of length 1. """
	
	# get the translation dictionnary from the file table.py
	table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
	}
	
	# algorithm
	# check that the sequence length is divisible by 3
		# loop over the sequence until the end
			# extract the next single codon
			# look up the codon and store the result
	
	protein = ""
	
	if len(seq) % 3 == 0:
		for i in range(0, len(seq), 3):
			codon = seq[i:i+3]
			protein += table[codon]
	return protein
#####
prt=read_seq('aminoacids.txt')
dna=read_seq('nucleotides.txt')

prt == translate(dna[20:938])[:-1] # returns True
# we omit the last codon 'stop'

