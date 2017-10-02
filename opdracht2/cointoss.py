import numpy as np

def cointoss(bitstrings):
	
	rho = np.zeros(len(bitstrings[0])) #list recording which 1 bits have been found

	for word in bitstrings:
		for j in np.arange(len(word)):
			if word[j] == '1':
				rho[j] = 1
				break
				
	print('Rho:', rho)
	
	#find up to which element in rho we have a 1 (so, find R)
	R = 0
	for r in rho:
		if r == 0:
			break
		R += 1
		
	print('R:', R)
	
	return 2**R
