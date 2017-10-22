import numpy as np
from time import time
bands = 10

def jaccard(row1, row2):
	length1 = len(row1)
	length2 = len(row2)
	union = len(np.union1d(row1, row2))
	intersect = length1+length2-union
	
	if union > 0:
		return intersect/union
	else: 
		return 0

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

	print("begin loop")
	start_time = time()
	# Hash bands to buckets #
	buckets = {}	
	for sigrow, i in zip(sigMatrix, np.arange(len(sigMatrix))):
		bands = np.split(np.array(sigrow), 5)
		for band in bands:
			buckets.setdefault(tuple(band),[]).append(i)
	
	print("Loop took %s seconds to execute" % (time() - start_time))
	
	true = 0
	false = 0
	for k in buckets.keys():
		if len(buckets[k]) > 1:
			for i in np.arange(len(buckets[k])):
				user1 = buckets[k][i]
				for j in np.arange(i+1, len(buckets[k])):
					user2 = buckets[k][j]
					if user1 != user2 and len(matrix[user1]) < 2*len(matrix[user2]) and len(matrix[user1]) > len(matrix[user2])/2:
						jaccardval = jaccard(matrix[user1], matrix[user2])
						if jaccardval > 0.5:
							print(user1, user2, len(buckets[k]), k, buckets[k])
							true += 1
						else:
							false += 1
	print(true, false)
	
