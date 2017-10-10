import numpy as np
import otherFunctions as oF

def prob_count(bitstrings, numbits, printprogress = True):

	print('\n--------- Cointoss ---------')
	
	rholist = np.zeros(numbits) #list recording which 1 bits have been found
	
	looplength = len(bitstrings)
	
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
		rholist[oF.rho(bitstrings[i], numbits)] = 1
	
	print('\n') #make space for new prints
	print('Rho:', rholist)
	
	#find up to which element in rho we have a 1 (so, find R)
	R = 0
	for r in rholist:
		if r == 0:
			break
		R += 1
		
	print('R:', R)
	
	return 2**R
