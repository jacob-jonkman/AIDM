import hashlib
import numpy as np

def rho(bitstring):
	counter = 1
	for bit in bitstring:
		if bit == "1":
			return counter
		counter += 1
	return counter-1


def loglogcount(bitstrings, k):
	buckets = 2**k
	M = np.zeros(buckets, dtype=int)
	alpha = 0.79402 # maar dan iets anders
	
	for bitstring in bitstrings:
		j = int(bitstring[:k], 2)
		r = rho(bitstring[k:])
		M[j] = max(M[j], r)
	
	exponent = 2**(sum(M) / buckets)
	return int(alpha * buckets * exponent)
