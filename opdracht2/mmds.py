import numpy as np
import otherFunctions as oF
import math

def mmds_count(bitstrings, numbits, printprogress = True):
	num_buckets = numbits
	partitions = 10
	bucket_len = int(np.ceil(len(bitstrings)/partitions))
	
	rhos = np.zeros((partitions+1, bucket_len), dtype=int)

	bucket = 0
	
	for i in np.arange(len(bitstrings)-1):
		bitstring = bitstrings[i]

		r = oF.rho(bitstring, numbits)
		rhos[bucket][i%bucket_len] = r
		print(bitstring, r)

		if i % bucket_len == 0:
			bucket += 1

	meanrhos = np.array([2**np.mean(rhos[:,i]) for i in np.arange(bucket_len)])
	print(meanrhos)
	return np.median(meanrhos)
