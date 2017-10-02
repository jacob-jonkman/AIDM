import numpy as np
import otherFunctions as oF

def cointoss(bitstrings, printprogress = True):

	print('\n--------- Cointoss ---------')
	
	rho = np.zeros(len(bitstrings[0])) #list recording which 1 bits have been found
	
	looplength = len(bitstrings)
	
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
		for j in np.arange(len(bitstrings[i])):
			if bitstrings[i][j] == '1':
				rho[j] = 1
				break
	
	print('\n') #make space for new prints
	print('Rho:', rho)
	
	#find up to which element in rho we have a 1 (so, find R)
	R = 0
	for r in rho:
		if r == 0:
			break
		R += 1
		
	print('R:', R)
	
	return 2**R
