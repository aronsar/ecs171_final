#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Importing tensorflow and numpy...")
import tensorflow as tf
import numpy as np
print("Imported")

print("Loading data...")
data = np.load('top20correlated.npy')
print("Data loaded")
train_N = int(data.shape[0] * .8)
test_N = data.shape[0] - train_N
dim = data.shape[1] - 1 # the -1 is b/c the last element is the label (ie loss)
num_epochs = 2000
print_every = 100
batch_size = 200
learning_rate = 5e-3
n_hidden1 = 100
n_hidden2 = 5

X_train = data[:train_N, 0:-1]
y_train = data[:train_N, -1] * 100 # the *100 is b/c data_sanitizer normalized the loss
X_test  = data[train_N:, 0:-1]
y_test  = data[train_N:, -1] * 100

X = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None])

std = 1
tf.set_random_seed(42)
np.random.seed(42)

weights = {
  'h1': tf.Variable(tf.random_normal([dim, n_hidden1], stddev=std)),
  'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=std)),
  'out': tf.Variable(tf.random_normal([n_hidden2, 1], stddev=std))
}

biases = {
  'b1': tf.Variable(tf.random_normal([n_hidden1], stddev=std)),
  'b2': tf.Variable(tf.random_normal([n_hidden2], stddev=std)),
  'out': tf.Variable(tf.random_normal([1], stddev=std))
}

#hidden1_out = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
#hidden2_out = tf.nn.relu(tf.add(tf.matmul(hidden1_out, weights['h2']), biases['b2']))
#prediction = tf.nn.relu(tf.add(tf.matmul(hidden2_out, weights['out']), biases['out']))
#X_print = tf.Print(X, [X, tf.reduce_mean(X)], message="X is: ", first_n=3, summarize=20)
#weights['h1'] = tf.Print(weights['h1'], [weights['h1'], tf.reduce_mean(weights['h1'])], message="weights['h1'] is: ", first_n=3, summarize=20)

#prediction = tf.maximum(tf.cast(0, tf.float32), tf.add(tf.matmul(X, weights['h1']), biases['b1']))
h1_out1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
h1_out2 = tf.nn.batch_normalization(h1_out1, tf.reduce_mean(h1_out1), 1, 0, 1, .0000001)
h1_out = tf.nn.relu(h1_out2)

h2_out1 = tf.add(tf.matmul(h1_out, weights['h2']), biases['b2'])
h2_out2 = tf.nn.batch_normalization(h2_out1, tf.reduce_mean(h2_out1), 1, 0, 1, .0000001)
h2_out = tf.nn.relu(h2_out2)

pred1 = tf.add(tf.matmul(h2_out, weights['out']), biases['out'])
pred2 = tf.nn.batch_normalization(pred1, tf.reduce_mean(pred1), 1, 0, 1, .0000001)
prediction = tf.nn.relu(pred2)
#prediction = tf.Print(prediction, [pred1, pred2, prediction], message="prediction is: ", first_n=3, summarize=20)

#y_print = tf.Print(y, [y, tf.shape(y)], message="y is: ", first_n=3, summarize=20)

denom = tf.cast(tf.shape(X)[0], tf.float32)
MAE = tf.reduce_sum(tf.abs(tf.expand_dims(y, 1) - prediction)) / denom
#MAE = tf.Print(MAE, [abs, tf.reduce_mean(abs), tf.reduce_max(abs), tf.shape(abs), sum, denom, MAE], message="MAE info: ")

optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(MAE)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

test_data = {X: X_test, y: y_test}

for i in range(num_epochs):
    idxs = np.random.choice(train_N, size=batch_size, replace=False)
    X_train_batch = X_train[idxs]
    y_train_batch = y_train[idxs]
    train_data = {X: X_train_batch, y: y_train_batch}
    
    # train step
    sess.run(train_step, feed_dict=train_data)
    
    if i % print_every == 0:
      # find train and test MAE of current model
      train_MAE = sess.run([MAE], feed_dict=train_data)
      test_MAE = sess.run([MAE], feed_dict=test_data)
      #print(train_MAE)
      #print(test_MAE)
      print("Train MAE: %.5f, Test MAE: %.5f" % (train_MAE[0], test_MAE[0]))
