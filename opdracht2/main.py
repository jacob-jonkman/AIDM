import numpy as np
import loglogcounting as llc
import cointoss as ct
from timeit import default_timer

numrows = 1000
numbits = 12

def counttrue(bitstrings):
	"""
	Count the true number of distinct elements
	"""
	unique = [0] #initialisation is needed with at least one value

	for word in bitstrings:
		for un in unique:
			if np.sum(unique == word) == 0:
				unique = np.append(unique, word)
	
	return len(unique) - 1 #decrease by one, because of the initialization
	
	
	return 0

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
