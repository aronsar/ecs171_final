#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

data = np.load('train_data_sanitized.npy')
train_N = int(data.shape[0] * .8)
test_N = data.shape[0] - train_N
dim = data.shape[1] - 1 # the -1 is b/c the last element is the label (ie loss)
num_epochs = 500
batch_size = 200
#mu = .9 # referred to as momentum, but basically equivalent to friction coefficient
learning_rate = .0000000002
n_hidden1 = 150
n_hidden2 = 50

X_train = data[:train_N, 0:-1]
y_train = data[:train_N, -1] * 100 # the *100 is b/c data_sanitizer normalized the loss
X_test  = data[train_N:, 0:-1]
y_test  = data[train_N:, -1] * 100
test_data = {X: X_test, y: y_test}

X = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, 1])

weights = {
  'h1': tf.Variable(tf.random_normal([dim, n_hidden1])),
  'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
  'out': tf.Variable(tf.random_normal([n_hidden2, 1]))
}

biases = {
  'b1': tf.Variable(tf.random_normal([n_hidden1])),
  'b2': tf.Variable(tf.random_normal([n_hidden2])),
  'out': tf.Variable(tf.random_normal([1]))
}

init = tf.global_variables_initializer()

hidden1_out = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
hidden2_out = tf.nn.relu(tf.add(tf.matmul(hidden1_out, weights['h2']), biases['b2']))
prediction = tf.add(tf.matmul(hidden2_out, weights['out']), biases['out'])

MAE = tf.sum(tf.abs(y - tf.cast(prediction, int32))) / batch_size
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(MAE)

sess = tf.Session()
sess.run(init)

for i in range(num_epochs):
    idxs = np.random.choice(train_N, size=batch_size, replace=False)
    X_train_batch = X_train[idxs]
    y_train_batch = y_train[idxs]
    train_data = {X: X_train_batch, y: y_train_batch}
    # test_data defined at the top of this file
    
    # train step
    sess.run(train_step, feed_dict=train_data)
    
    if i % 10 == 0:
      # find train and test MAE of current model
      train_MAE = sess.run([MAE], feed_dict=train_data)
      test_MAE = sess.run([MAE], feed_dict=test_data)
      print("Train MAE: %.5f, Test MAE: %.5f" % (train_MAE, test_MAE))