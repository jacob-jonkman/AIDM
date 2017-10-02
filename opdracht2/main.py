import numpy as np
import loglogcounting as llc
import cointoss as ct

numrows = 1000
numbits = 10

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
	
	truecount = counttrue(bitstrings)
	print('The true amount of unique elements:', truecount)
	
	#llcount = llc.loglogcount(bitstrings)
	ctcount = ct.cointoss(bitstrings)
	print("Cointoss count:", ctcount)
	
	#print(bitstrings)

if __name__ == "__main__":
	main()
