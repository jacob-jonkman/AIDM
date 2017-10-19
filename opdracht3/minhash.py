import numpy as np

bands = 100

def min_hash(matrix, num_movies, num_users, num_hashes):
	A = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	B = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	c = 2999
	
	sigMatrix = []
	for row in matrix[1:]:
		signature = []
		
		# For each row, find the signatures for all the different hashes #
		# Append them to sigMatrix afterwards #
		for i in np.arange(num_hashes):
			hashcodes = np.array([(A[i]*row + B[i]) % c])
			signature.append(np.min(hashcodes))
		
		sigMatrix.append(signature)
	
	# sigMatrix bevat nu voor alle users die we hebben bekeken de signature row. #
	#print(sigMatrix)
	#print(len(sigMatrix))
	
	###
	# Permute M randomly by
	# p= randperm
	# M[p,:]
	###
	
