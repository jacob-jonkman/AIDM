import numpy as np
import minhash_writelogs as mh

from time import time
#import truejaccard as tj

num_hashes = 100

def main():
	## TODO: Command line parse ##
	start_time = time()
	raw_data = np.load("user_movie.npy")
	
	np.random.seed(42)
	
	f = open('run_test.txt', 'w')
	#f.write("Program took %s seconds to execute\n" % (time() - start_time))
	f.write("Run test succesfull!")
	f.close()
	
	# Find number of users and movies. Each user and movie ID actually occurs #
	num_users = np.max(raw_data[:,0])
	num_movies = np.max(raw_data[:,1])
	print(num_users, num_movies)
	
	# Count the number of occurrences of each user and keep in a list 	#
	# This is used to efficiently fill the user,movie matrix 						#
	user_counts = np.bincount(raw_data[:,0])
	max_user_count = np.max(user_counts)
	print(user_counts, max_user_count)

	# Do the same for the movies. This is not used yet #
	movie_counts = np.bincount(raw_data[:,1])	
	max_movie_count = np.max(movie_counts)
	print(movie_counts)
	#m = coo_matrix(raw_data)
	#matrix = m.tocsr()
	
	# The user,movie matrix to be filled
	matrix = np.array([np.zeros(user_counts[i]) for i in np.arange(num_users)])
	
	# Fill the user,movie matrix
	user_start = 0
	for user in np.arange(1,num_users):
		user_count = user_counts[user]
		user_end = user_start+user_count
		
		# Slice the raw data to contain all user,movie pairs of the current user #
		# and put this in the user,movie matrix #
		row = raw_data[user_start:user_end,1]
		matrix[user] = row
		
		# Compute the start position of the next user in the raw data #
		user_start += user_count
		
		#print(user, matrix[user].shape)

	mh.min_hash(matrix, num_movies, num_users, 20)
	
	#tj.findTrueJac(matrix)
	
	f = open('runtime_log.txt', 'w')
	print("Program took %s seconds to execute" % (time() - start_time))
	
	f.write("Program took %s seconds to execute" % (time() - start_time))
	f.close()

if __name__ == "__main__":
	main()
