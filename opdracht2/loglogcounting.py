import numpy as np
import otherFunctions as oF

def rho(bitstring, numbits):
	if bitstring == 0:
		return numbits
	p = 0
	while (bitstring >> p) & 1 == 0:
		p += 1
	return p


def loglogcount(bitstrings, k, numbits, printprogress = True):

	print('\n--------- LogLog Count ---------')

	buckets = 2**k
	M = np.zeros(buckets, dtype=int)
	alpha = 0.79402
	
	looplength = len(bitstrings)
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
		j = bitstrings[i] & (buckets-1)
		buckethash = bitstrings[i] >> k
		r = rho(buckethash, numbits)
		M[j] = max(M[j], r)
	
	exponent = 2**(sum(M) / buckets)
	
	print('\n')
	
	return int(alpha * buckets * exponent)
