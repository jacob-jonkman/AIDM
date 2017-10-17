def min_hash(matrix, num_movies):
	A = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	B = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	c = 2999
	
	###
	# Permute M randomly by
	# p= randperm
	# M[p,:]
	###
