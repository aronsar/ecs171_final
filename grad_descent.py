from __future__ import division
import numpy as np
import time
import pdb

data = np.load('train_data_sanitized.npy')
train_N = int(data.shape[0] * .8)
test_N = data.shape[0] - train_N
dim = data.shape[1] - 1
numEpochs = 200
learning_rate = .00000000002

#pdb.set_trace()

X_train = data[:train_N, 0:-1]
y_train = data[:train_N, -1] * 100
X_test  = data[train_N:, 0:-1]
y_test  = data[train_N:, -1] * 100

weights = np.zeros(dim)
gradient = np.zeros(dim)
loss = 0

start = time.time()

print("all zeros MAE: %.4f" % (np.sum(np.abs(y_train)) / train_N))

for t in range(numEpochs):
  gradient = np.sum(np.sign(np.matmul(X_train, weights.T) - y_train)[:, np.newaxis] * X_train)
  # gradient calc goes here (770 x 1)
  
  weights = weights - learning_rate * gradient
  
  if (t % 10 == 0):
    prediction = np.matmul(X_train, weights.T)
    train_MAE = np.sum(np.abs(y_train - prediction)) / train_N # this is loss
    test_MAE = np.sum(np.abs(y_test - np.matmul(X_test, weights.T))) / test_N
    # loss calc goes here
    
    print("Epoch: %d, train MAE: %.4f, test MAE: %.4f" % (t, train_MAE, test_MAE))
  #import pdb; pdb.set_trace()


end = time.time()
print("Running time was %.2f seconds." % (end - start))