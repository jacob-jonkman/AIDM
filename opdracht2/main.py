import numpy as np
import loglogcounting as llc
import prob_count as pt
from timeit import default_timer
import otherFunctions as oF
import mmds
import sys #for writing print statements to a log file
import matplotlib.pyplot as plt
import seaborn as sns

#plot with the seaborn visual defaults
sns.set()

#put all print statements in the log file. Set to false if you want only an
#incomplete log, but get output in the terminal
completelog = False
if completelog:
	sys.stdout = open("Log.txt", "w")

def counttrue(bitstrings, numrows, numbits, printprogress = True):
	"""
	Count the true number of distinct elements.
	"""
	if printprogress:
		print('\n--------- True Count ---------')
	
	#array containing whether or not each integer exists in the bitstrings
	stringsfound = np.zeros(2**numbits, dtype = bool)

	#for displaying the progress
	looplength = len(bitstrings)
	
	for i in np.arange(looplength):
		if printprogress:
			oF.progress(i, looplength)
			
		stringsfound[bitstrings[i] - 1] = True
	
	if printprogress:
		print('\n') #make space for new prints
	
	return np.count_nonzero(stringsfound)
	
def runCounts(numrows, numbits, doprints = True):
	#try to load the database of bitstrings. If this file is not present, create the data and save it to file
	try:
		bitstrings = np.load("./Bitstrings_archive/bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits))
		if doprints:
			print("Loaded bitstrings from file...")
	except:
		if doprints:
			print("Creating new bitstrings...")
		bitstrings = np.random.randint(2**numbits, size=(numrows))
		
		#only save the data if it is not too large
		if numrows * numbits < 5e7:
			if doprints:
				print('Saving bitstrings to file...')
			np.save("./Bitstrings_archive/bitstrings_rows={0}_bits={1}.npy".format(numrows, numbits), bitstrings)
		elif doprints:
			print('Not saving bitstrings to file -> size limit reached')
	
	#open the log
	if completelog == False:
		log = open("Log.txt", "w")
		log.write("\n\n------- New log -------\n")
		log.write("numrows = {0}, numbits = {1}\n".format(numrows, numbits))
	
	#normal counting
	starttime = default_timer()
	truecount = counttrue(bitstrings, numrows, numbits, printprogress = doprints)
	tc_runtime = default_timer() - starttime
	
	if doprints:
		print('The true amount of unique elements: {0}'.format(truecount))
		print('Runtime: {0} seconds'.format(round(tc_runtime, 3)))
	if completelog == False:
		log.write('The true amount of unique elements: {0}\n'.format(truecount))
		log.write('Runtime: {0} seconds\n'.format(round(tc_runtime, 3)))
	
	#loglog counting
	starttime = default_timer()
	llcount = llc.loglogcount(bitstrings, 10, numbits, printprogress = doprints)
	ll_RAE = np.abs(truecount - llcount)/truecount * 100.
	ll_runtime = default_timer() - starttime
	
	#TEMP
	print("\nLoglog count: {0}, RAE: {1}%".format(llcount, round(ll_RAE, 3)))
	
	if doprints:
		print("Loglog count: {0}, RAE: {1}%".format(llcount, round(ll_RAE, 3)))
		print('Runtime: {0} seconds'.format(round(ll_runtime, 3)))
	if completelog == False:
		log.write("Loglog count: {0}, RAE: {1}%\n".format(llcount, round(ll_RAE, 3)))
		log.write('Runtime: {0} seconds\n'.format(round(ll_runtime, 3)))
	
	#probabilistic counting
	starttime = default_timer()
	ptcount = pt.prob_count(bitstrings, numbits, printprogress = doprints)
	pt_RAE = np.abs(truecount - ptcount)/truecount * 100.
	pt_runtime = default_timer() - starttime
	
	if doprints:
		print("Probabilistic count: {0}, RAE: {1}%".format(ptcount, round(pt_RAE, 3)))
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
	
	
def plotTwoValues(x, y1, y2, xlabel, y1label, y2label, title, filename):
	"""
	Makes a scatter plot of two sets of data with two y axes
	"""
	import matplotlib.ticker as tkr
	
	plt.close()
	plt.rc('text', usetex=True)
	#change plot style, add spines
	#sns.set_style("white")
	sns.set_style("ticks")
	sns.despine(top = True) 
	
	#plot with two axes:
	fig, ax1 = plt.subplots()
	#makes another y-axis
	ax2 = ax1.twinx()
	#plots first line
	lns1 = ax1.scatter(x, y1, alpha = 0.3, color = 'r')
	#set the ytick labels for this first y axis to scientific notation
	if np.max(y1) > 1e4:
		ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		
	#plots second line
	lns2 = ax2.scatter(x, y2, alpha = 0.3, color = 'b')
	#Changes the colour of the tick labels to match the line colour
	for tl in ax1.get_yticklabels():
		tl.set_color('r')
	for tl in ax2.get_yticklabels():
		tl.set_color('b')
	
	#disable grid -> not nice with two y axes
	ax1.grid(False)
	ax2.grid(False)
	
	#set the xtick labels to scientific notation
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	
	#increase left spacing 
	fig = plt.gcf()
	fig.subplots_adjust(left = 0.11, right = 0.9)
		
	ax1.set_ylabel(y1label)
	ax2.set_ylabel(y2label)
	ax1.set_xlabel(xlabel)
	plt.title(title)
	plt.savefig('./Figures/' + filename, dpi = 200)
	
def plotDifferentSettings():
	"""
	Make plots of the RAE and the runtime for different settings.
	
	This code could be made parallel using:
	https://stackoverflow.com/questions/8329974/can-i-get-a-return-value-from-multiprocessing-process#8330339
	"""
	#save location of the results
	resloc = 'Different_settings_results'

	#the numrows limits
	nrows_lims = [1e4, 1e7]
	nbits = 30
	
	#string for the file names of the to be saved files
	settingsstr = 'nrows={:.0e}--{:.0e}_nbits={}'.format(nrows_lims[0], nrows_lims[1], nbits)
	
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
	numrows = np.linspace(nrows_lims[0], nrows_lims[1], num = 15, dtype = int)
	numbits = np.array([nbits])
	
	looplength = len(numrows)
	
	try:
		(ll_RAE, prob_RAE, comb_RAE, tc_runtime, ll_runtime, prob_runtime, comb_runtime) = np.loadtxt('./{0}/diffset_results_{1}.txt'.format(resloc, settingsstr))
	except:
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
			
		np.savetxt('./{0}/diffset_results_{1}.txt'.format(resloc, settingsstr), 
			np.array([ll_RAE, prob_RAE, comb_RAE, tc_runtime, ll_runtime, prob_runtime, comb_runtime]), 
			header = '#ll_RAE, prob_RAE, comb_RAE, tc_runtime, ll_runtime, prob_runtime, comb_runtime')
	
	plotTwoValues(numrows, ll_RAE, ll_runtime, 'Number of rows', 'RAE [\%]', 'Runtime [s]', 'RAE and runtime of loglog count for different number of rows. \nNumbits = {}'.format(nbits), 'RAEandRuntime_loglog_{0}.pdf'.format(settingsstr))
	
	plotTwoValues(numrows, prob_RAE, prob_runtime, 'Number of rows', 'RAE [\%]', 'Runtime [s]', 'RAE and runtime of probabilisic count for different \nnumber of rows. Numbits = {}'.format(nbits), 'RAEandRuntime_prob_{0}.pdf'.format(settingsstr))


def plotInSingleFig():
	import seaborn as sns
	import pandas as pd
	
	plt.close()
	
	#save location of the results
	resloc = 'Different_settings_results'
	
	#the numrows limits
	nrows_lims = [1e4, 6e6]
	numrows = np.linspace(nrows_lims[0], nrows_lims[1], num = 15, dtype = int)
	#number of bits
	numbits = np.array([20, 25, 30])
	
	#the data frame where everything will be stored
	data = pd.DataFrame()
	
	#loop over the different number of bins for which the results were already calculated
	for nbit in numbits:
		#string for the save files=
		settingsstr = 'nrows={:.0e}--{:.0e}_nbits={}'.format(nrows_lims[0], nrows_lims[1], nbit)
	
		#load the data from a single run
		(ll_RAE, prob_RAE, comb_RAE, tc_runtime, ll_runtime, prob_runtime, comb_runtime) = np.loadtxt('./{0}/diffset_results_{1}.txt'.format(resloc, settingsstr))
	
		#create new local data frame
		#df = pd.DataFrame({'ll_RAE':ll_RAE, 'prob_RAE':prob_RAE, 'comb_RAE':comb_RAE, 
		#				'tc_runtime':tc_runtime, 'll_runtime':ll_runtime, 'prob_runtime':prob_runtime, 
		#				'comb_runtime':comb_runtime, 'numrows':numrows, 'numbits':np.full(len(numrows), nbit)})
		'''
		df = pd.DataFrame({'RAE':np.append(ll_RAE, prob_RAE), 
						'Runtime':np.append(ll_runtime, prob_runtime), 
						'numrows':np.tile(numrows, 2), 'numbits':np.full(len(numrows) * 2, nbit),
						'Algorithm':np.append(np.full(len(numrows), 'LogLog', dtype = str), 
						np.full(len(numrows), 'Probabilistic', dtype = str))})
		'''	
		df = pd.DataFrame({'RAE [\%]':np.append(ll_RAE, prob_RAE), 
						'Runtime':np.append(ll_runtime, prob_runtime), 
						'numrows':np.tile(numrows, 2), 'numbits':np.full(len(numrows) * 2, nbit),
						'Algorithm':np.append(np.tile(['LogLog'], len(numrows)), 
						np.tile(['Probabilistic'], len(numrows)))})
	
		#append the local data frame to the big one
		data = data.append(df, ignore_index = True)
	
		# Set style of scatterplot
	plt.rc('text', usetex=True)
	sns.set_context("notebook", font_scale=1.1)
	sns.set_style("ticks", {'axes.grid' : True})
		
	'''
	fig, axs = plt.subplots(ncols=2)
	sns.regplot(x='numrows', y='ll_RAE', data=df, fit_reg = False, ax=axs[0])
	sns.regplot(x='numrows', y='prob_RAE', data=df, fit_reg = False, ax=axs[0])
	#sns.regplot(x='value', y='wage', data=df_melt, ax=axs[1])
	#sns.boxplot(x='education',y='wage', data=df_melt, ax=axs[2])
	'''

	g = sns.FacetGrid(data, row="numbits", col = 'Algorithm', sharey = 'none', sharex = 'col', margin_titles=True) 
	g = (g.map(plt.scatter, "numrows", "RAE [\%]")).fig.subplots_adjust(wspace=.25, hspace=.05)
	
	fig = plt.gcf()
	#change overall size
	fig.set_size_inches(7.5, 9)
	#decrease spacing on the left
	fig.subplots_adjust(left = 0.12)
	
	#set the xtick labels to scientific notation
	#ax = plt.gca()
	#ax.get_xaxis().get_major_formatter().set_scientific(True)
	
	filename = 'All_settings_RAE.pdf'
	plt.savefig('./Figures/' + filename, dpi = 200, bbox_inches = 'tight')
	plt.show()

def main():
	np.set_printoptions(threshold=np.nan)
	
	np.random.seed(42)
	
	#Run the different count algorithms on a single data set
	#results = runCounts(int(1000000), 30)
	
	#plot the results of the count algorithms for different settings
	#plotDifferentSettings()
	
	plotInSingleFig()
	

if __name__ == "__main__":
	main()
