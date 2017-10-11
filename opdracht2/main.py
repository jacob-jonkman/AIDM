import numpy as np
import loglogcounting as llc
import prob_count as pt
from timeit import default_timer
import otherFunctions as oF
import mmds
import sys #for writing print statements to a log file
import matplotlib.pyplot as plt

#put all print statements in the log file. Set to false if you want only an
#incomplete log, but get output in the terminal
completelog = False
if completelog:
	sys.stdout = open("Log.txt", "w")

def counttrue(bitstrings, numrows, numbits, printprogress = True):
	"""
	Count the true number of distinct elements.
	"""
	print('\n--------- True Count ---------')
	
	#array containing whether or not each integer exists in the bitstrings
	stringsfound = np.zeros(2**numbits, dtype = bool)

	#for displaying the progress
	looplength = len(bitstrings)
	
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
			
		stringsfound[bitstrings[i] - 1] = True
	
	print('\n') #make space for new prints
	
	return np.count_nonzero(stringsfound)
	
def runCounts(numrows, numbits, doprints = True):
	#try to load the database of bitstrings. If this file is not present, create the data and save it to file
	try:
		bitstrings = np.load("./Bitstrings_archive/bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits))
		print("Loaded bitstrings from file...")
	except:
		print("Creating new bitstrings and saving them to file...")
		bitstrings = np.random.randint(2**numbits, size=(numrows))
		
		#only save the data if it is not too large
		if numrows * numbits < 5e7:
			np.save("./Bitstrings_archive/bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits), bitstrings)
	
	#open the log
	if completelog == False:
		log = open("Log.txt", "w")
		log.write("\n\n------- New log -------\n")
		log.write("numrows = {0}, numbits = {1}\n".format(numrows, numbits))
	
	#normal counting
	starttime = default_timer()
	truecount = counttrue(bitstrings, numrows, numbits, printprogress = doprints)
	print('The true amount of unique elements: {0}'.format(truecount))
	tc_runtime = default_timer() - starttime
	print('Runtime: {0} seconds'.format(round(tc_runtime, 3)))
	if completelog == False:
		log.write('The true amount of unique elements: {0}\n'.format(truecount))
		log.write('Runtime: {0} seconds\n'.format(round(tc_runtime, 3)))
	
	#loglog counting
	starttime = default_timer()
	llcount = llc.loglogcount(bitstrings, 10, numbits, printprogress = doprints)
	ll_RAE = np.abs(truecount - llcount)/truecount * 100.
	print("Loglog count: {0}, RAE: {1}%".format(llcount, round(ll_RAE, 3)))
	ll_runtime = default_timer() - starttime
	print('Runtime: {0} seconds'.format(round(ll_runtime, 3)))
	if completelog == False:
		log.write("Loglog count: {0}, RAE: {1}%\n".format(llcount, round(ll_RAE, 3)))
		log.write('Runtime: {0} seconds\n'.format(round(ll_runtime, 3)))
	
	#probabilistic counting
	starttime = default_timer()
	ptcount = pt.prob_count(bitstrings, numbits, printprogress = doprints)
	pt_RAE = np.abs(truecount - ptcount)/truecount * 100.
	print("Probabilistic count: {0}, RAE: {1}%".format(ptcount, round(pt_RAE, 3)))
	pt_runtime = default_timer() - starttime
	print('Runtime: {0} seconds'.format(round(pt_runtime, 3)))
	if completelog == False:
		log.write("Probabilistic count: {0}, RAE: {1}%\n".format(ptcount, round(pt_RAE, 3)))
		log.write('Runtime: {0} seconds\n'.format(round(pt_runtime, 3)))
		
		
	mmds_RAE = -1.
	mmds_runtime = -1.
	'''
	#combinatorial counting
	starttime = default_timer()
	mmdscount = mmds.mmds_count(bitstrings, numbits, printprogress=True)
	mmds_RAE = np.abs(truecount - mmdscount)/truecount * 100.
	print("Combinatorial count: {0}, RAE: {1}%".format(mmdscount, round(mmds_RAE, 3)))
	mmds_runtime = default_timer() - starttime
	print('Runtime: {0} seconds'.format(round(mmds_runtime, 3)))
	if completelog == False:
		log.write("Combinatorial count: {0}, RAE: {1}%\n".format(mmdscount, round(mmds_RAE, 3)))
		log.write('Runtime: {0} seconds\n'.format(round(mmds_runtime, 3)))
	'''
	if completelog == False:
		log.close()
		
	return np.array([ll_RAE, pt_RAE, mmds_RAE, tc_runtime, ll_runtime, pt_runtime, mmds_runtime])
	
def plotDifferentSettings():
	"""
	Make plots of the RAE and the runtime for different settings
	"""
	#the relative approximation error for the different counting algorithms
	ll_RAE = []
	prob_RAE = []
	comb_RAE = []
	#the runtime for the different algorithms
	tc_runtime = []
	ll_runtime = []
	prob_runtime = []
	comb_runtime = []

	#the different settings we want to test
	numrows = np.linspace(1e4, 1e7, num = 15, dtype = int)
	numbits = np.array([20])
	
	looplength = len(numrows)
	
	for i in np.arange(len(numrows)):
		oF.progress(i, looplength)
		for j in np.arange(len(numbits)):
			results = runCounts(numrows[i], numbits[j], doprints = False)
			
			ll_RAE = np.append(ll_RAE, results[0])
			prob_RAE = np.append(prob_RAE, results[1])
			comb_RAE = np.append(comb_RAE, results[2])
			
			tc_runtime = np.append(tc_runtime, results[3])
			ll_runtime = np.append(ll_runtime, results[4])
			prob_runtime = np.append(prob_runtime, results[5])
			comb_runtime = np.append(comb_runtime, results[6])
			
	np.savetxt('./diffsettings_results.txt', np.array([ll_RAE, prob_RAE, comb_RAE, tc_runtime, ll_runtime, prob_runtime, comb_runtime]), header = '#ll_RAE, prob_RAE, comb_RAE, tc_runtime, ll_runtime, prob_runtime, comb_runtime')
			
	plt.scatter(numrows, ll_RAE, alpha = 0.3)
	plt.xlabel('Number of rows')
	plt.ylabel('RAE')
	plt.title('RAE of loglog count for different number of rows')
	plt.savefig('./Figures/RAE_loglog.png', dpi = 200)
	

def main():
	np.set_printoptions(threshold=np.nan)
	
	np.random.seed(42)
	
	#Run the different count algorithms on a single data set
	#results = runCounts(int(1000000), 30)
	
	#plot the results of the count algorithms for different settings
	plotDifferentSettings()
	

if __name__ == "__main__":
	main()
