import numpy as np

maxRating = 5
minRating = 1
folds = 5

def naive_global(ratings):
	return np.mean(ratings)


def naive_item(data, num_movies):
	itemratings = np.zeros(num_movies)
	itemcounts = np.zeros(num_movies)
	
	for i in np.arange(len(data)):
		print(i)
		itemratings[data[i,1] - 1] += data[i,2]
		itemcounts[data[i,1] - 1] += 1
	
	return np.round(itemratings/itemcounts, 1)


# Compute mean rating per user
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


def count_distinct(lst):
	return np.max(lst)


def main():
	data = np.genfromtxt("ml-1m/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int', max_rows=100000)
	
	data[:,2] = roundRatings(data[:,2])
	
	

	sum_naive_global = 0
	sum_naive_item = 0
	sum_naive_user = 0
	sum_naive_user_item = 0
	
	num_users = count_distinct(data[:,0])
	num_movies = count_distinct(data[:,1])
	
	for fold in range(folds):
		train_set = np.array([data[x] for x in range(len(data)) if (x%5) != fold])
		test_set = np.array([data[x] for x in range(len(data)) if (x%5) == fold])
		
		sum_naive_global += naive_global(train_set[:,2])
		sum_naive_user += naive_user(train_set, num_users)
		sum_naive_item += naive_item(train_set, num_movies)
		
	print("mean of naive global classifier: ", sum_naive_global/folds)
	print("means of naive user classifier: ", sum_naive_user/folds)
	print("means of naive item classifier: ", sum_naive_item/folds)
		
		
	
if __name__ == "__main__":
	main()
