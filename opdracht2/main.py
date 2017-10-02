import numpy as np
import loglogcounting as llc
import cointoss as ct
from timeit import default_timer
import otherFunctions as oF

numrows = int(1e6)
numbits = 20

def bitstringsToInt(bitstrings):
	"""
	Converts a bitstring array to an int array
	"""
	intarray = np.zeros(len(bitstrings), dtype = int)
	for i in np.arange(len(bitstrings)):
		intarray[i] = int(bitstrings[i], 2)
		
	return intarray

def counttrue(bitstrings, printprogress = True):
	"""
	Count the true number of distinct elements.
	"""
	print('\n--------- True Count ---------')
	
	#convert to integers to speed up the counting
	intarray = bitstringsToInt(bitstrings)
	
	#array containing the frequency of each integer
	intfreq = np.zeros(2**numbits, dtype = int)
	
	#for displaying the progress
	looplength = len(intarray)
	
	for i in np.arange(len(intarray)):
		if printprogress:
			oF.progress(i, looplength)
			
		intfreq[intarray[i] - 1] += 1
	
	print('\n') #make space for new prints
	
	return np.count_nonzero(intfreq)

def main():
	np.set_printoptions(threshold=np.nan)
	
	#try to load the database of bitstrings. If this file is not present, create the data and save it to file
	try:
		bitstrings = np.load("./bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits))
		print("Loaded bitstrings from file...")
	except:
		print("Creating new bitstrings and saving them to file...")
		bitarray = np.random.randint(2, size=(numrows, numbits))
		bitstrings = np.array([str(e).replace(' ','').replace('[','').replace(']','') for e in bitarray])
		np.save("./bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits), bitstrings)
	
		#normal counting
	starttime = default_timer()
	truecount = counttrue(bitstrings, printprogress = True)
	print('The true amount of unique elements:', truecount)
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))
	
	#llcount = llc.loglogcount(bitstrings)
		#cointoss
	starttime = default_timer()
	ctcount = ct.cointoss(bitstrings, printprogress = True)
	print("Cointoss count: {0}, Error: {1}%".format(ctcount, round(np.abs(truecount - ctcount)/truecount * 100., 3)))
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))

if __name__ == "__main__":
	main()
