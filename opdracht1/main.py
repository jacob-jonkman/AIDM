import numpy as np

maxRating = 5
minRating = 1
folds = 5

def naive_global(ratings):
	return np.mean(ratings)


def naive_item(data):
	pass


def naive_user(data, num_users):
	userratings = np.zeros(num_users)
	usercounts = np.zeros(num_users)
	i=0
	while(i < len(data)):
		userratings[data[i,0]-1] += data[i,2]
		usercounts[data[i,0]-1] += 1
		i+=1
		
	return np.round(userratings/usercounts, 1)


def naive_user_item(user, item, data):
	pass


## Squeeze ratings into range 1 to 5
def roundRatings(ratings):
	return(np.array([max(min(x, maxRating), minRating) for x in ratings]))


def count_distinct_users(data):
	return len(np.unique(data))


def count_distinct_movies(data):
	return len(np.unique(data))


def main():
	data = np.genfromtxt("ml-1m/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int', max_rows=1000)
	
	data[:,2] = roundRatings(data[:,2])

	sum_naive_global = 0
	sum_naive_item = 0
	sum_naive_user = 0
	sum_naive_user_item = 0
	
	num_users = count_distinct_users(data[:,0])
	num_movies = count_distinct_movies(data[:,1])
	
	for fold in range(folds):
		train_set = np.array([data[x] for x in range(len(data)) if (x%5) != fold])
		test_set = np.array([data[x] for x in range(len(data)) if (x%5) == fold])
		
		sum_naive_global += naive_global(train_set[:,2])
		sum_naive_user += naive_user(train_set, num_users)
		sum_naive_item = naive_item(train_set)
		
	print("mean of naive global classifier: ", sum_naive_global/5)
	print("means of naive user classifier: ", sum_naive_user/5)
		
		
	
if __name__ == "__main__":
	main()
