import hashlib
import numpy as np
import otherFunctions as oF

def rho(bitstring):
	counter = 1
	for bit in bitstring:
		if bit == "1":
			return counter
		counter += 1
	return counter-1


def loglogcount(bitstrings, k, printprogress = True):

	print('\n--------- LogLog Count ---------')

	buckets = 2**k
	M = np.zeros(buckets, dtype=int)
	alpha = 0.79402 # maar dan iets anders
	
	looplength = len(bitstrings)
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
		j = int(bitstrings[i][:k], 2)
		r = rho(bitstrings[i][k:])
		M[j] = max(M[j], r)
	
	exponent = 2**(sum(M) / buckets)
	
	print('\n')
	
	return int(alpha * buckets * exponent)
