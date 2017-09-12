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
		itemratings[data[i,1] - 1] += data[i,2]
		itemcounts[data[i,1] - 1] += 1
	
	return np.round(itemratings/itemcounts, 1)


# Compute mean rating per user
def naive_user(data, num_users):
	userratings = np.zeros(num_users)
	usercounts = np.zeros(num_users)

	for i in range(len(data)):
		userratings[data[i,0]-1] += data[i,2]
		usercounts[data[i,0]-1] += 1
		
	return np.round(userratings/usercounts, 1)


def naive_model(data, num_users, num_movies):#user, item):
	def model(X, a, b, c):
		x1, x2 = X
		return a*x1 + b*x2 + c
		
	#Can be used to make a fit to a model
	from scipy.optimize import curve_fit
	#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
	#returns the best values for the parameters of the model in the array popt
	#the array pcov contains the estimated covariance of popt
	#p0 are the values for the variables it will start to look 
	
	x1 = naive_user(train_set, num_users)
	x2 = naive_item(train_set, num_movies)
	
	print("hier!!!!!")
	print(x1, x2)
	
	popt, pcov = curve_fit(model, (x1, x2), ydata, p0 = (1000., 1.))
	#Then the standard deviation is given by:
	perr = np.sqrt(np.diag(pcov))
	#get the residuals:
	residuals = ydata- model(xdata, popt)
	#to get R^2:
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((ydata-np.mean(ydata))**2)
	r_squared = 1 - (ss_res / ss_tot)
	
	


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
	sum_naive_model = 0
	
	num_users = count_distinct(data[:,0])
	num_movies = count_distinct(data[:,1])
	
	for fold in range(folds):
		train_set = np.array([data[x] for x in range(len(data)) if (x%5) != fold])
		test_set = np.array([data[x] for x in range(len(data)) if (x%5) == fold])
		
		sum_naive_global += naive_global(train_set[:,2])
		sum_naive_user += naive_user(train_set, num_users)
		sum_naive_item += naive_item(train_set, num_movies)
		sum_naive_model += naive_model(train_set, num_users, num_movies)
		
	print("mean of naive global classifier: ", sum_naive_global/folds)
	print("means of naive user classifier: ", sum_naive_user/folds)
	print("means of naive item classifier: ", sum_naive_item/folds)
		
		
	
if __name__ == "__main__":
	main()
