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
	print("users: ", num_users, "items:", num_movies)
		
	global_average_ratings = np.zeros(folds)
	naive_user_ratings = np.zeros(num_users)
	naive_item_ratings = np.zeros(num_movies)
	
	# Apply 5-fold cross validation
	for fold in np.arange(folds):
		print("fold", fold, end="")
		
		train_set = np.array([data[x] for x in np.arange(len(data)) if (x%5) != fold])
		test_set = np.array([data[x] for x in np.arange(len(data)) if (x%5) == fold])
		
		global_average_ratings[fold] = naive_global(train_set[:,2])
		naive_user_ratings += naive_user(train_set, num_users, global_average_ratings[fold])
		naive_item_ratings += naive_item(train_set, num_movies, global_average_ratings[fold])
		
		print(" done")
		
	naive_user_ratings /= folds
	naive_item_ratings /= folds
	
	# Construct lin_reg_data = [avg for user, avg for movie]
	lin_reg_data = np.zeros((len(data), 2))
	for i in range(len(data)):
		lin_reg_data[i,0] = naive_user_ratings[data[i,0]-1]
		lin_reg_data[i,1] = naive_item_ratings[data[i,1]-1]
	
	regression_coeffs = naive_model(lin_reg_data, data[:,2])
	print("Linear Regression done. Coefficients:", regression_coeffs)


if __name__ == "__main__":
	main()
