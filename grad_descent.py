from __future__ import division
import numpy as np
import time
import pdb

data = np.load('train_data_sanitized.npy')
train_N = int(data.shape[0] * .8)
test_N = data.shape[0] - train_N
dim = data.shape[1] - 1 # the -1 is b/c the last element is the label (ie loss)
numEpochs = 500
batch_size = 50
mu = .9 # referred to as momentum, but basically equivalent to friction coefficient
learning_rate = .0000000002

X_train = data[:train_N, 0:-1]
y_train = data[:train_N, -1] * 100 # the *100 is b/c data_sanitizer normalized the loss
X_test  = data[train_N:, 0:-1]
y_test  = data[train_N:, -1] * 100

#pdb.set_trace()
weights = np.random.normal(size=dim)
gradient = np.zeros(dim)
v = np.zeros(dim)
loss = 0

start = time.time()

print("all zeros MAE: %.4f" % (np.sum(np.abs(y_train)) / train_N))

for t in range(numEpochs):
  idxs = np.random.choice(train_N, size=batch_size, replace=False)
  X_train_batch = X_train[idxs]
  y_train_batch = y_train[idxs]
  
  gradient = np.sum(np.sign(np.matmul(X_train_batch, weights.T) - y_train_batch)[:, np.newaxis] * X_train_batch)
  # gradient calc goes here (770 x 1)
  v = mu * v - learning_rate * gradient
  weights = weights + v
  
  if (t % 50 == 0):
    prediction = np.matmul(X_train_batch, weights.T)
    train_MAE = np.sum(np.abs(y_train_batch - prediction.astype(int))) / batch_size # this is loss
    test_MAE = np.sum(np.abs(y_test - np.matmul(X_test, weights.T).astype(int))) / test_N
    # loss calc goes here
    
    print("Epoch: %d, train MAE: %.4f, test MAE: %.4f" % (t, train_MAE, test_MAE))
  #import pdb; pdb.set_trace()


end = time.time()
print("Running time was %.2f seconds." % (end - start))