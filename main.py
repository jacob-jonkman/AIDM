import numpy as np
import pandas as pd

maxRating = 5
minRating = 1
folds = 5

def naive_global(ratings):
  return np.mean(ratings)

def naive_item(user, item, data):
  pass

def naive_user(user, item, data):
  pass

def naive_user_item(user, item, data):
  pass

## Squeeze ratings into range 1 to 5
def roundRatings(ratings):
  return(np.array([max(min(x, maxRating), minRating) for x in ratings]))

def main():
  data = np.genfromtxt("ml-1m/ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int',  max_rows=1000)
  data[:,2] = roundRatings(data[:,2])
  print(data[:,2])
  
  for fold in range(folds):
    train_set = np.array([data[x] for x in range(len(data)) if (x%5) != fold])
    test_set = np.array([data[x] for x in range(len(data)) if (x%5) == fold])
    #train_set = np.array([data[x] if (x%5) != fold for x in range(len(data))])
    #test_set =  np.array([data[x] if (x%5) == fold for x in range(len(data))])
    print(naive_global(data[:,2]))
  #print(naive_user(data[:,2:3))
  
  
if __name__ == "__main__":
  main()
