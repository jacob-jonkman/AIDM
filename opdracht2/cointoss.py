import numpy as np

def cointoss(bitstring):
	
	rho = np.zeros(len(bitstring[0])) #rank of the first bit which is 1 in the bitstring

	for bit in bitstring:
		for j in np.arange(len(bit)):
			if bit[j] and rho == j - 1:
				rho += 1
				break
	print(rho)
	return 10**rho
