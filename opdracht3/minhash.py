import numpy as np

bands = 10

def jaccard(matrix, user1, user2):
	union = len(np.union1d(matrix[user1], matrix[user2]))
	intersect = len(np.intersect1d(matrix[user1], matrix[user2]))
	return intersect/union

def dohash(row, num_hashes, bandwidth = 4):
	return(np.sum(np.array([row[i:i+bandwidth]])**2) for i in np.arange(0, num_hashes, bandwidth))

def min_hash(matrix, num_movies, num_users, num_hashes):
	A = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	B = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	c = 2999
	
	"""sigMatrix = np.zeros(0)
	
	for i in np.arange(num_hashes):
		permutation = np.random.permutation(matrix)
		signatures = np.array([(A[i]*row + B[i]) % c for row in matrix])
		sigMatrix = np.concatenate((sigMatrix, signatures))
	
	sigMatrix = np.reshape(sigMatrix, (num_users, num_hashes))
	print("na reshape", sigMatrix.shape)
	
	print(np.min(sigMatrix))
	
	"""
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
	print(len(sigMatrix))
	
	buckets = {key: [] for key in np.arange(2**20)}
	
	for row, i in zip(sigMatrix, np.arange(len(sigMatrix))):
		hashed_row = dohash(row, num_hashes)
		for band in hashed_row:
			if(buckets[band]):
				buckets[band].append(i)
			else:
				buckets[band] = [i]
	true = 0
	false = 0
	for k in buckets.keys():
		if len(buckets[k]) > 1:
			for i in np.arange(len(buckets[k])):
				for j in np.arange(i+1, len(buckets[k])):
					user1 = buckets[k][i]
					user2 = buckets[k][j]
					jaccardval = jaccard(matrix, user1, user2)
					if jaccardval > 0.5:
						print(user1, user2)
						true += 1
					else:
						false += 1
	print(true, false)

