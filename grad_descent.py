from __future__ import division
import numpy as np
import time

N = 50000
dim = 770
numEpochs = 200
learning_rate = .00005

train = np.load('train_data_sanitized.npy')
import pdb; pdb.set_trace()

X_train = train[:, :-2]
y_train = train[:, -1]

weights = np.zeros(dim)
gradient = np.zeros(dim)
loss = 0

start = time.time()

for t in range(numEpochs):
  gradient = -np.sum(X_train, axis=0)
  # gradient calc goes here (770 x 1)
  
  weights = weights - learning_rate * gradient
  
  if (t % 1 == 0):
    prediction = np.matmul(X_train, weights.T)
    MAE = np.sum(y_train - prediction) / N # this is loss
    # loss calc goes here
    
    print("MAE at epoch %d = %.3f" % (t, MAE))
  #import pdb; pdb.set_trace()


end = time.time()
print("Running time was %.2f seconds." % (end - start))