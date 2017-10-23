import numpy as np
import minhash as mh
import argparse
from time import time
from scipy.sparse import coo_matrix

def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('seed', type=int)
	parser.add_argument('datapath', type=str)
	args = parser.parse_args()
	
	start_time = time()
	raw_data = np.load(args.datapath)
	
	#clear the results file
	with open("./results.txt", "w"): pass
	
	np.random.seed(args.seed)
	
	# Find number of users and movies. Each user and movie ID actually occurs #
	num_users = np.max(raw_data[:,0])
	num_movies = np.max(raw_data[:,1])
	
	# Count the number of occurrences of each user and keep in a list 	#
	# This is used to efficiently fill the user,movie matrix 						#
	user_counts = np.bincount(raw_data[:,0])
	max_user_count = np.max(user_counts)

	# Do the same for the movies #
	movie_counts = np.bincount(raw_data[:,1])	
	max_movie_count = np.max(movie_counts)
	
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

	mh.min_hash(matrix, num_movies, num_users, 72, num_bands=12)
	
	print("Program took %s seconds to execute" % (time() - start_time))

if __name__ == "__main__":
	main()
