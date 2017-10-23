import numpy as np
from time import time

def jaccard(row1, row2):
	length1 = len(row1)
	length2 = len(row2)
	union = len(np.union1d(row1, row2))
	intersect = length1+length2-union
	
	if union > 0:
		return intersect/union
	else: 
		return 0

def min_hash(matrix, num_movies, num_users, num_hashes, num_bands = 10):
	"""
		Determines which users in a user-movie data have a jaccard similarity of at 
		least 0.5. Saves results (user user pairs) to 'results.txt'.
	
		Input:
			matrix (int, 2d numpy array): the matrix containing the users and the movies 
			which they rated.\n
			num_movies (int): the total amount of unique movies.\n
			num_users (int): the total amount of unique users.\n
			num_hashes (int): the number of hashes there will be used for the minhashing
			algorithm.\n
			num_bands (int): the number of bands that the signature of a user will be 
			split in.
	"""
	#the variables needed for the minhashing
	A = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	B = np.array([np.random.randint(0, num_movies) for i in np.arange(num_hashes)])
	c = 2999
	
	results = []
	
	sigMatrix = []
	for row in matrix[1:]:
		signature = []
		
		# For each row, find the signatures for all the different hashes #
		# Append them to sigMatrix afterwards #
		for i in np.arange(num_hashes):
			hashcodes = np.array([(A[i]*row + B[i]) % c])
			signature.append(np.min(hashcodes))
		
		sigMatrix.append(signature)

	# Hash bands to buckets #
	buckets = {}	
	for sigrow, i in zip(sigMatrix, np.arange(len(sigMatrix))):
		bands = np.split(np.array(sigrow), num_bands)
		for band in bands:
			buckets.setdefault(tuple(band),[]).append(i)
	
	#the count of users that had a jaccard similarity larger than 0.5 and those 
	#lower than 0.5
	true = 0
	
	#loop over all the buckets
	for k in buckets.keys():
		if len(buckets[k]) > 1: #if there are two or more users in the bucket
			#loop over the users in the bucket
			for i in np.arange(len(buckets[k])):
				user1 = buckets[k][i]
				#loop over all the users 
				for j in np.arange(i+1, len(buckets[k])):
					user2 = buckets[k][j]
					if user1 != user2 and len(matrix[user1]) < 2*len(matrix[user2]) and len(matrix[user1]) > len(matrix[user2])/2:
						jaccardval = jaccard(matrix[user1], matrix[user2])
						if jaccardval > 0.5:
							if (user1,user2) not in results:
								print(user1, user2)
								results.append((user1,user2))
								true += 1
								
								#write the output to file 
								with open("./results.txt", "a") as f:
									f.write("{0}, {1}\n".format(user1, user2))
							else:
								print("dubbel!")	
	print(true, "matches found")
	
