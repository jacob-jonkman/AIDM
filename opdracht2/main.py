import numpy as np
import loglogcounting as llc
import cointoss as ct
from timeit import default_timer
import otherFunctions as oF
import sys #for writing print statements to a log file

numrows = int(1e6)
numbits = 30

#put all print statements in the log file. Set to false if you want only an
#incomplete log, but get output in the terminal
completelog = False
if completelog:
	sys.stdout = open("Log.txt", "w")

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
		
	#array containing whether or not each integer exists in the bitstrings
	intfreq = np.zeros(2**numbits, dtype = bool)

	#for displaying the progress
	looplength = len(bitstrings)
	
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
			
		intfreq[int(bitstrings[i], 2) - 1] = True
	
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
		
		#only save the data if it is not too large
		if numrows * numbits < 5e7:
			np.save("./bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits), bitstrings)
	
	#open the log
	if completelog == False:
		log = open("Log.txt", "w")
		log.write("\n\n------- New log -------\n")
		log.write("numrows = {0}, numbits = {1}\n".format(numrows, numbits))
	
	#normal counting
	starttime = default_timer()
	truecount = counttrue(bitstrings, printprogress = True)
	print('The true amount of unique elements: {0}'.format(truecount))
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))
	if completelog == False:
		log.write('The true amount of unique elements: {0}\n'.format(truecount))
		log.write('Runtime: {0} seconds\n'.format(round(default_timer() - starttime, 3)))
	
	#loglog counting
	starttime = default_timer()
	llcount = llc.loglogcount(bitstrings, 10)
	print("Loglog count: {0}, Error: {1}%".format(llcount, round(np.abs(truecount - llcount)/truecount * 100., 3)))
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))
	if completelog == False:
		log.write("Loglog count: {0}, Error: {1}%\n".format(llcount, round(np.abs(truecount - llcount)/truecount * 100., 3)))
		log.write('Runtime: {0} seconds\n'.format(round(default_timer() - starttime, 3)))
	
	#cointoss
	starttime = default_timer()
	ctcount = ct.cointoss(bitstrings, printprogress = True)
	print("Cointoss count: {0}, Error: {1}%".format(ctcount, round(np.abs(truecount - ctcount)/truecount * 100., 3)))
	print('Runtime: {0} seconds'.format(round(default_timer() - starttime, 3)))
	if completelog == False:
		log.write("Cointoss count: {0}, Error: {1}%\n".format(ctcount, round(np.abs(truecount - ctcount)/truecount * 100., 3)))
		log.write('Runtime: {0} seconds\n'.format(round(default_timer() - starttime, 3)))
	
	if completelog == False:
		log.close()
	

if __name__ == "__main__":
	main()
