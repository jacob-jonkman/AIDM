import numpy as np
#import matplotlib.pyplot as plt

def jaq(row1, row2):
	intersect = len(np.intersect1d(row1, row2))
	union = len(np.union1d(row1, row2))
	
	return intersect/union

def findTrueJac(matrix):
	reducedMat = matrix[:1000]
	
	try:
		jaqvalues = np.load('jaqdata.npy')
	except:
		jaqvalues = []
	
		for i in np.arange(len(reducedMat)):
			for j in np.arange(i+1, len(reducedMat)):
				jv = jaq(reducedMat[i], reducedMat[j])
				jaqvalues = np.append(jaqvalues, jv)
		
				if jv > 0.3:
					print(jv)
				
		np.save('jaqdata.npy', jaqvalues)
	
	print('> 0.5:', len(jaqvalues > 0.5))
	print('> 0.4:', len(jaqvalues > 0.4))
	print('Total tests:', len(jaqvalues))
