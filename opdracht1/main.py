import numpy as np
from sklearn import linear_model

maxRating = 5
minRating = 1
folds = 5

# Compute the mean of all ratings in the data
def naive_global(ratings):
	return np.mean(ratings)


# Compute for each movie what the average rating of that movie is across all users
def naive_item(data, num_movies, global_average_rating):
	itemratings = np.zeros(num_movies)
	itemcounts = np.zeros(num_movies)
	
	for i in np.arange(len(data)):
		itemratings[data[i,1] - 1] += data[i,2]
		itemcounts[data[i,1] - 1] += 1
		
	return np.round([itemratings[j]/itemcounts[j] if itemcounts[j] > 0 else global_average_rating for j in range(num_movies)], 1)


# Compute for each user what their average given rating is across all rated movies
def naive_user(data, num_users, global_average_rating):
	userratings = np.zeros(num_users)
	usercounts = np.zeros(num_users)

	for i in np.arange(len(data)):
		userratings[data[i,0]-1] += data[i,2]
		usercounts[data[i,0]-1] += 1
	
	return np.round([userratings[j]/usercounts[j] if usercounts[j] > 0 else global_average_rating for j in range(num_users)], 1)


# Use linear regression
def naive_model(lin_reg_data, ratings):
	reg = linear_model.LinearRegression()
	reg.fit(lin_reg_data, ratings)	
	return reg.coef_
	

## Squeeze ratings into range 1 to 5
def roundRatings(ratings):
	return(np.array([max(min(x, maxRating), minRating) for x in ratings]))


def main():
	## data = [userID, movieID, rating] ##
	data = np.genfromtxt("ml-1m/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')
	
	# Compute the number of users and movies in the data. This assumes that every movieID 
	# and every userID lower than the biggest one in the data occurs at least once.
	num_users = np.max(data[:,0])
	num_movies = np.max(data[:,1])
		
	# Initialize lists to store the average ratings per movie and user
	user_ratings = np.zeros(num_users)
	item_ratings = np.zeros(num_movies)
	
	# Lists to store the errors over the folds
	err_train_gar = np.zeros(folds)
	err_train_urt = np.zeros(folds)
	err_train_irt = np.zeros(folds)
	err_test_gar = np.zeros(folds)
	err_test_urt = np.zeros(folds)
	err_test_irt = np.zeros(folds)
	
	np.random.seed(17)
	
	# Apply 5-fold cross validation
	for fold in np.arange(folds):
		np.random.shuffle(data)
		
		print("Start fold", fold)
		train_set = np.array([data[x] for x in np.arange(len(data)) if (x % folds) != fold])
		test_set = np.array([data[x] for x in np.arange(len(data)) if (x % folds) == fold])
		
		# Compute the naive classifiers on the train set
		global_average_rating = naive_global(train_set[:,2])
		user_ratings = naive_user(train_set, num_users, global_average_rating)
		item_ratings = naive_item(train_set, num_movies, global_average_rating)
		
		# Construct avg_ratings_list_train = [avg for user, avg for movie] for all data in train set
		avg_ratings_list = np.zeros((len(train_set), 2))
		for i in np.arange(len(train_set)):
			avg_ratings_list[i,0] = user_ratings[data[i,0]-1]
			avg_ratings_list[i,1] = item_ratings[data[i,1]-1]
			
		# Construct avg_ratings_list_train = [avg for user, avg for movie] for all data in train set
		avg_ratings_list_test = np.zeros((len(test_set), 2))
		for i in np.arange(len(test_set)):
			avg_ratings_list_test[i,0] = user_ratings[data[i,0]-1]
			avg_ratings_list_test[i,1] = item_ratings[data[i,1]-1]
		
		# apply the naive models to the train set and compute errors
		err_train_gar[fold] = np.sqrt(np.mean((train_set[:,2] - global_average_rating)**2))
		err_train_urt[fold] = np.sqrt(np.mean((train_set[:,2] - avg_ratings_list[:,0])**2))
		err_train_irt[fold] = np.sqrt(np.mean((train_set[:,2] - avg_ratings_list[:,1])**2))
		
		# apply the naive models to the test set and compute errors
		err_test_gar[fold]	= np.sqrt(np.mean((test_set[:,2] - global_average_rating)**2))
		err_test_urt[fold]	= np.sqrt(np.mean((test_set[:,2] - avg_ratings_list_test[:,0])**2))
		err_test_irt[fold]	= np.sqrt(np.mean((test_set[:,2] - avg_ratings_list_test[:,1])**2))
	
		# Print errors, both for the train and the test set
		print("errors on train set:", err_train_gar[fold], err_train_urt[fold], err_train_irt[fold])
		print("errors on test set:", err_test_gar[fold], err_test_urt[fold], err_test_irt[fold])
		
		# Apply Linear Regression
		regression_coeffs = naive_model(avg_ratings_list, train_set[:,2])
		print("Linear Regression done. Coefficients:", regression_coeffs)
		
		print()

if __name__ == "__main__":
	main()
