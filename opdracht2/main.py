import numpy as np
import loglogcounting as llc
import cointoss as ct

numrows = 10
numbits = 20

def counttrue(bitstrings):
	"""
	Count the true number of distinct elements
	"""
	#list of all unique elements
	unique = [bitstrings[0], bitstrings[1]]
	print(unique)
	for word in bitstrings:
		for un in unique:
			if np.sum(unique == word) == 0:
				unique = np.append(unique, word)
	
	print(unique)
	
	return len(unique)
	
	
	return 0

def main():
	np.set_printoptions(threshold=np.nan)
	bitarray = np.random.randint(2, size=(numrows, numbits))
	bitstrings = np.array([str(e).replace(' ','').replace('[','').replace(']','') for e in bitarray])
	
	truecount = counttrue(bitstrings)
	print('The true amount of unique elements:', truecount)
	
	#llcount = llc.loglogcount(bitstrings)
	#ctcount = ct.cointoss(bitstrings)
	
	#print(bitstrings)

if __name__ == "__main__":
	main()
