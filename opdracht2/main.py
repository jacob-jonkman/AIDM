import numpy as np
import loglogcounting as llc
import cointoss as ct

numrows = 1000
numbits = 20

def main():
	np.set_printoptions(threshold=np.nan)
	bitarray = np.random.randint(2, size=(numrows, numbits))
	bitstrings = np.array([str(e).replace(' ','').replace('[','').replace(']','') for e in bitarray])
	
	llcount = llc.loglogcount(bitstrings)
	#cointoss(bitstrings)
	
	print(bitstrings)

if __name__ == "__main__":
	main()
