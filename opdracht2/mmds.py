import numpy as np
import otherFunctions as oF
import math

def mmds_count(bitstrings, numbits, printprogress = True):
	if printprogress:
		print('\n--------- Combinatorial ---------')

	num_buckets = numbits
	partitions = 10
	bucket_len = int(np.ceil(len(bitstrings)/partitions))
	
	rhos = np.zeros((partitions+1, bucket_len), dtype=int)

	bucket = 0
	
	looplength = len(bitstrings) - 1
	
	for i in np.arange(len(bitstrings)-1):
		if printprogress:
			oF.progress(i, looplength)
		bitstring = bitstrings[i]

		r = oF.rho(bitstring, numbits)
		rhos[bucket][i%bucket_len] = r

		if i % bucket_len == 0:
			bucket += 1

	meanrhos = np.array([2**np.mean(rhos[:,i]) for i in np.arange(bucket_len)])
	
	if printprogress:
		print('\n')
	
	return np.median(meanrhos)
