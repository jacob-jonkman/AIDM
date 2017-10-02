import numpy as np
import loglogcounting as llc
import cointoss as ct
from timeit import default_timer
import otherFunctions as oF

numrows = 2000
numbits = 14

def counttrue(bitstrings):
	"""
	Count the true number of distinct elements.
	"""
	print('\n--------- True Count ---------')
	unique = [0] #initialisation is needed with at least one value
	
	#for displaying the progress
	looplength = len(bitstrings)
	
	for i in np.arange(len(bitstrings)):
		oF.progress(i, looplength)
			
		for un in unique:
			if np.sum(unique == bitstrings[i]) == 0:
				unique = np.append(unique, bitstrings[i])
	
	print('\n') #make space for new prints
	
	return len(unique) - 1 #decrease by one, because of the initialization

def main():
	np.set_printoptions(threshold=np.nan)
	bitarray = np.random.randint(2, size=(numrows, numbits))
	bitstrings = np.array([str(e).replace(' ','').replace('[','').replace(']','') for e in bitarray])
	
		#normal counting
	starttime = default_timer()
	truecount = counttrue(bitstrings)
	print('The true amount of unique elements:', truecount)
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))
	
	#llcount = llc.loglogcount(bitstrings)
		#cointoss
	starttime = default_timer()
	ctcount = ct.cointoss(bitstrings)
	print("Cointoss count:", ctcount)
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))
	
	#print(bitstrings)

if __name__ == "__main__":
	main()
