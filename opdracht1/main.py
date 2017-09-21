import numpy as np
from sklearn import linear_model
from multiprocessing import Pool

maxRating = 5
minRating = 1
folds = 5

num_factors = 40
num_iter = 75
regularization = 0.005
learn_rate = 0.005

num_users = 0
num_movies = 0

naive_errors_train = np.zeros((folds, 3))
naive_errors_test  = np.zeros((folds, 3))

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


# Compute Linear Regression Coefficients
def linearRegression(lin_reg_data, ratings):
	reg = linear_model.LinearRegression()
	reg.fit(lin_reg_data, ratings)	
	return reg.coef_, reg.intercept_
	

## Squeeze ratings into range 1 to 5
def roundRatings(ratings):
	return(np.array([max(min(x, maxRating), minRating) for x in ratings]))


def applyNaiveModels(data, user_ratings, item_ratings, avg_ratings_list, global_average_rating):
	errors = np.zeros(3)
	
	# Construct avg_ratings_list_train = [avg for user, avg for movie] for the passed data
	for i in np.arange(len(data)):
		avg_ratings_list[i,0] = user_ratings[data[i,0]-1]
		avg_ratings_list[i,1] = item_ratings[data[i,1]-1]
	
	# Apply the naive models to the data and compute errors
	errors[0] = np.sqrt(np.mean((data[:,2] - global_average_rating)**2))
	errors[1] = np.sqrt(np.mean((data[:,2] - avg_ratings_list[:,0])**2))
	errors[2] = np.sqrt(np.mean((data[:,2] - avg_ratings_list[:,1])**2))
	
	return errors, avg_ratings_list

def Xmatrix(data, num_users, num_movies):
  # Convert the data set to the IxJ matrix  
  X = np.zeros((num_users, num_movies)) * np.nan
  for i in np.arange(len(data)):
    X[data[i,0]-1,data[i,1]-1] = data[i,2]
    
  return X

def matrixFact(data, num_users, num_movies, fold):
	# Convert the data set to the IxJ matrix  
	X_data = Xmatrix(data, num_users, num_movies)

	X_hat = np.zeros(num_users, num_movies) #The matrix of predicted ratings
	E = np.zeros(num_users, num_movies) #The error values 
	
	# The matrices used to determine the ratings. These are initialized with random values and then converged to optimum values using gradient descent.
	U = np.random.rand(num_users, num_factors) 
	M = np.random.rand(num_factors, num_movies)
	U_prime = U
	M_prime = M
  
	for q in np.arange(num_iter):
		for i in np.arange(len(data)):
			userId = data[i,0] - 1
			movieId = data[i,1] - 1
			actual = data[i,2]			
			prediction = np.sum(U[userId,:] * M[:,movieId])
			error = actual - prediction
			
			# Update U and M by building U_prime and M_prime, which will replace U and M when done with this iteration
			for k in np.arange(num_factors):
				U_prime[userId, k]  = U[userId, k]  + learn_rate * (2 * error * M[k, movieId] - regularization * U[userId, k])
				M_prime[k, movieId] = M[k, movieId] + learn_rate * (2 * error * U[userId, k]  - regularization * M[k, movieId])

		U = U_prime
		M = M_prime
		
		# Compute intermediate MSE
		X_hat = np.dot(U,M)
		E = X_data - X_hat
		intermediate_error = np.sqrt(np.mean(E[np.where(np.isnan(E) == False)]**2))
		
		print("Fold", fold, "Iteration", q, "out of", num_iter, "done. Error:", intermediate_error)
	
	# Apply U and M one last time and return the result
	X_hat = np.dot(U,M)
	return X_hat
  
  
def doFold(fold):
	global data
	global num_users
	global num_movies

	# Shuffle the data and divide in train and test set
	np.random.shuffle(data)
	train_set = np.array([data[x] for x in np.arange(len(data)) if (x % folds) != fold])
	test_set = np.array([data[x] for x in np.arange(len(data)) if (x % folds) == fold])

	# Initialize lists to store the average ratings per movie and user, as well as the errors per model per fold
	user_ratings = np.zeros(num_users)
	item_ratings = np.zeros(num_movies)

	# Compute the naive classifiers on the train set
	global_average_rating = naive_global(train_set[:,2])
	user_ratings = naive_user(train_set, num_users, global_average_rating)
	item_ratings = naive_item(train_set, num_movies, global_average_rating)

	avg_ratings_list_train = np.zeros((len(train_set), 2))
	avg_ratings_list_test = np.zeros((len(test_set), 2))

	# Apply the models on the train and test set
	naive_errors_train[fold,:], avg_ratings_list_train = applyNaiveModels(train_set, user_ratings, item_ratings, avg_ratings_list_train, global_average_rating)
	naive_errors_test[fold,:], avg_ratings_list_test = applyNaiveModels(test_set, user_ratings, item_ratings, avg_ratings_list_test, global_average_rating)

	# Print errors, both for the train and the test set
	print("Fold", fold, "errors on train set:", naive_errors_train[fold, 0], naive_errors_train[fold, 1], naive_errors_train[fold, 2])
	print("Fold", fold, "errors on test set:", naive_errors_test[fold, 0], naive_errors_test[fold, 1], naive_errors_test[fold, 2])

	# Apply Linear Regression
	regr_coeffs, regr_intercept = linearRegression(avg_ratings_list_train, train_set[:,2])
	regression_predictions_train = roundRatings(regr_coeffs[0]*avg_ratings_list_train[:,0] + regr_coeffs[1]*avg_ratings_list_train[:,1] + regr_intercept)
	regression_predictions_test = roundRatings(regr_coeffs[0]*avg_ratings_list_test[:,0] + regr_coeffs[1]*avg_ratings_list_test[:,1] + regr_intercept)

	regr_error_train = np.sqrt(np.mean((train_set[:,2] - regression_predictions_train)**2))
	regr_error_test = np.sqrt(np.mean((test_set[:,2] - regression_predictions_test)**2))

	print("Fold", fold, "Linear Regression done. Coefficients:", regr_coeffs, regr_intercept)
	print("Fold", fold, "Training error using linear regression:", regr_error_train)
	print("Fold", fold, "Test error using linear regression:", regr_error_test)

	X_hat = matrixFact(train_set, num_users, num_movies, fold)
	X_train = Xmatrix(train_set, num_users, num_movies)
	X_test = Xmatrix(test_set, num_users, num_movies)

	E_train = X_train - X_hat
	E_test = X_test - X_hat

	MF_error_train = np.sqrt(np.mean(E_train[np.where(np.isnan(E_train) == False)]**2))
	MF_error_test = np.sqrt(np.mean(E_test[np.where(np.isnan(E_test) == False)]**2))

	print("Fold", fold, 'MF training set error:', MF_error_train)
	print("Fold", fold, 'MF test set error:', MF_error_test)

def main():
	global data
	global num_users
	global num_movies

	## Read dataset into data with format [userID, movieID, rating]
	data = np.genfromtxt("ml-1m/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int', max_rows = 10000)
	
	# Compute the number of users and movies in the data. This assumes that every movieID 
	# and every userID lower than the biggest one in the data occurs at least once.
	num_users = np.max(data[:,0])
	num_movies = np.max(data[:,1])
	
	np.random.seed(17)
	pool = Pool()
	pool.map(doFold, range(folds))
	"""
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
		naive_errors_train[fold,:], avg_ratings_list_train = applyNaiveModels(train_set, user_ratings, item_ratings, avg_ratings_list_train, global_average_rating)
		naive_errors_test[fold,:], avg_ratings_list_test = applyNaiveModels(test_set, user_ratings, item_ratings, avg_ratings_list_test, global_average_rating)
		
		# Print errors, both for the train and the test set
		print("errors on train set:", naive_errors_train[fold, 0], naive_errors_train[fold, 1], naive_errors_train[fold, 2])
		print("errors on test set:", naive_errors_test[fold, 0], naive_errors_test[fold, 1], naive_errors_test[fold, 2])
		
		# Apply Linear Regression
		regr_coeffs, regr_intercept = linearRegression(avg_ratings_list_train, train_set[:,2])
		regression_predictions_train = roundRatings(regr_coeffs[0]*avg_ratings_list_train[:,0] + regr_coeffs[1]*avg_ratings_list_train[:,1] + regr_intercept)
		regression_predictions_test = roundRatings(regr_coeffs[0]*avg_ratings_list_test[:,0] + regr_coeffs[1]*avg_ratings_list_test[:,1] + regr_intercept)
		
		regr_error_train = np.sqrt(np.mean((train_set[:,2] - regression_predictions_train)**2))
		regr_error_test = np.sqrt(np.mean((test_set[:,2] - regression_predictions_test)**2))
		
		print("Linear Regression done. Coefficients:", regr_coeffs, regr_intercept)
		print("Training error:", regr_error_train)
		print("Test error:", regr_error_test)
		
		
		X_hat = matrixFact(train_set, num_users, num_movies)
		X_train = Xmatrix(train_set, num_users, num_movies)
		X_test = Xmatrix(test_set, num_users, num_movies)
		
		E_train = X_train - X_hat
		E_test = X_test - X_hat
		
		MF_error_train = np.sqrt(np.mean(E_train[np.where(np.isnan(E_train) == False)]**2))
		MF_error_test = np.sqrt(np.mean(E_test[np.where(np.isnan(E_test) == False)]**2))
		
		print('MF training set error:', MF_error_train)
		print('MF test set error:', MF_error_test)
"""

if __name__ == "__main__":
	main()
