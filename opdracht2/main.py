import numpy as np
import loglogcounting
import cointoss

numrows = 1000
numbits = 20

def main():
	np.set_printoptions(threshold=np.nan)
	bitarray = np.random.randint(2, size=(numrows, numbits))
	bitstrings = ''.join(str(e).replace(' ','') for e in bitarray)
	
	loglogcount(bitstrings)
	cointoss(bitstrings)
	
	print(bitstrings)

if __name__ == "__main__":
	main()
