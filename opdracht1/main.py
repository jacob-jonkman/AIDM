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


def applyNaiveModels(data, user_ratings, item_ratings, avg_ratings_list, global_average_rating):
	errors = np.zeros(3)
	
	# Construct avg_ratings_list_train = [avg for user, avg for movie] for all data in train set
	for i in np.arange(len(data)):
		avg_ratings_list[i,0] = user_ratings[data[i,0]-1]
		avg_ratings_list[i,1] = item_ratings[data[i,1]-1]
	
	# apply the naive models to the train set and compute errors
	errors[0] = np.sqrt(np.mean((data[:,2] - global_average_rating)**2))
	errors[1] = np.sqrt(np.mean((data[:,2] - avg_ratings_list[:,0])**2))
	errors[2] = np.sqrt(np.mean((data[:,2] - avg_ratings_list[:,1])**2))
	
	return errors, avg_ratings_list

def main():
	## Read dataset into data with format [userID, movieID, rating]
	data = np.genfromtxt("ml-1m/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')
	
	# Compute the number of users and movies in the data. This assumes that every movieID 
	# and every userID lower than the biggest one in the data occurs at least once.
	num_users = np.max(data[:,0])
	num_movies = np.max(data[:,1])
		
	# Initialize lists to store the average ratings per movie and user, as well as the errors per model per fold
	user_ratings = np.zeros(num_users)
	item_ratings = np.zeros(num_movies)
	
	errors_train = np.zeros((folds, 3))
	errors_test  = np.zeros((folds, 3))
	
	np.random.seed(17)
	
	# Apply 5-fold cross validation
	for fold in np.arange(folds):
		print("Start fold", fold)
		
		# Shuffle the data and divide in train and test set
		np.random.shuffle(data)
		train_set = np.array([data[x] for x in np.arange(len(data)) if (x % folds) != fold])
		test_set = np.array([data[x] for x in np.arange(len(data)) if (x % folds) == fold])
		
		# Compute the naive classifiers on the train set
		global_average_rating = naive_global(train_set[:,2])
		user_ratings = naive_user(train_set, num_users, global_average_rating)
		item_ratings = naive_item(train_set, num_movies, global_average_rating)
		
		avg_ratings_list_train = np.zeros((len(train_set), 2))
		avg_ratings_list_test = np.zeros((len(test_set), 2))
		
		# Apply the models on the train and test set
		errors_train[fold,:], avg_ratings_list_train = applyNaiveModels(train_set, user_ratings, item_ratings, avg_ratings_list_train, global_average_rating)
		errors_test[fold,:], avg_ratings_list_test = applyNaiveModels(test_set, user_ratings, item_ratings, avg_ratings_list_test, global_average_rating)
		
		# Print errors, both for the train and the test set
		print("errors on train set:", errors_train[fold, 0], errors_train[fold, 1], errors_train[fold, 2])
		print("errors on test set:", errors_test[fold, 0], errors_test[fold, 1], errors_test[fold, 2])
		
		# Apply Linear Regression
		regression_coeffs = naive_model(avg_ratings_list_train, train_set[:,2])
		print("Linear Regression done. Coefficients:", regression_coeffs)
		
		print()

if __name__ == "__main__":
	main()
