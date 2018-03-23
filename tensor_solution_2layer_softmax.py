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
num_epochs = 200
print_every = 20
batch_size = 200
learning_rate = 5e-3
n_hidden1 = 15
n_hidden2 = 7 # output of this is the logits layer

X_train = data[:train_N, 0:-1]
y_train = data[:train_N, -1] * 11 # the *100 is b/c data_sanitizer normalized the loss
X_test  = data[train_N:, 0:-1]
y_test  = data[train_N:, -1] * 11

X = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None])

std = 1
tf.set_random_seed(42)
np.random.seed(42)

weights = {
  'h1': tf.Variable(tf.random_normal([dim, n_hidden1], stddev=std)),
  'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=std))
}

biases = {
  'b1': tf.Variable(tf.random_normal([n_hidden1], stddev=std)),
  'b2': tf.Variable(tf.random_normal([n_hidden2], stddev=std))
}

labels = tf.constant([0, 1, 2, 3, 5, 7, 11])

#weights['h1'] = tf.Print(weights['h1'], [weights['h1'], biases['b1']], message="weights and biases: ")

#hidden1_out = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
#hidden2_out = tf.nn.relu(tf.add(tf.matmul(hidden1_out, weights['h2']), biases['b2']))
#prediction = tf.nn.relu(tf.add(tf.matmul(hidden2_out, weights['out']), biases['out']))
#X_print = tf.Print(X, [X, tf.reduce_mean(X)], message="X is: ", first_n=3, summarize=20)
#weights['h1'] = tf.Print(weights['h1'], [weights['h1'], tf.reduce_mean(weights['h1'])], message="weights['h1'] is: ", first_n=3, summarize=20)

#prediction = tf.maximum(tf.cast(0, tf.float32), tf.add(tf.matmul(X, weights['h1']), biases['b1']))
h1_out1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
h1_mean, h1_variance = tf.nn.moments(h1_out1, axes=1)
h1_out2 = tf.nn.batch_normalization(h1_out1, tf.expand_dims(h1_mean, 1), tf.expand_dims(h1_variance, 1), 0, 1, .0000001)
h1_out = tf.nn.relu(h1_out2)

h2_out1 = tf.add(tf.matmul(h1_out, weights['h2']), biases['b2'])
h2_mean, h2_variance = tf.nn.moments(h2_out1, axes=1)
h2_out2 = tf.nn.batch_normalization(h2_out1, tf.expand_dims(h2_mean, 1), tf.expand_dims(h2_variance, 1), 0, 1, .0000001)
#h2_out = tf.nn.relu(h2_out2) # logits layer
h2_out = h2_out2
'''
h2_out_print = tf.Print(h2_out, [tf.reduce_mean(h2_out1), 
  tf.reduce_min(h2_out1),
  tf.reduce_max(h2_out1),
  tf.reduce_mean(h2_out2),
  tf.reduce_min(h2_out2),
  tf.reduce_max(h2_out2),
  h2_out2], message="h2_out: ", summarize=20)
'''
probabilities = tf.nn.softmax(h2_out)
#inside_stuff = tf.expand_dims(y, 1) * tf.log(probabilities)
#cat_idx1 = tf.stack([tf.range(0, tf.shape(probabilities)[0]), tf.cast(y, tf.int32)], axis=1)
idxs1 = tf.where(tf.transpose(tf.equal(tf.cast(y, tf.int32), tf.expand_dims(labels, 1))))
log_likelihood = -tf.log(tf.gather_nd(probabilities, idxs1))
cross_entropy = tf.reduce_mean(log_likelihood)
#cross_entropy_print = tf.Print(h2_out2, [h2_out2], message="cross_entropy: ", summarize=10)
#probabilities = tf.Print(probabilities, [tf.shape(probabilities)], message="probabilities :")
is_correct = tf.equal(tf.cast(y, tf.int32), tf.gather(labels, tf.arg_max(probabilities, 1)))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#prediction = tf.Print(prediction, [pred1, pred2, prediction], message="prediction is: ", first_n=3, summarize=20)

#y_print = tf.Print(y, [y, tf.shape(y)], message="y is: ", first_n=3, summarize=20)

denom = tf.cast(tf.shape(X)[0], tf.float32)
prediction = tf.cast(tf.arg_max(probabilities, 1), tf.float32)
#prediction = tf.Print(prediction, [prediction], message="prediction: ", summarize=20)
abs = tf.abs(y - prediction)
sum = tf.reduce_sum(abs)
MAE = sum / denom

#MAE = tf.Print(MAE, [prediction, tf.reduce_max(prediction), tf.reduce_mean(prediction), cross_entropy], message="cross_entropy: ", summarize=100)


# MAE = tf.Print(MAE, [#h1_out1,
                     # #h2_out, 
                     # #tf.reduce_min(prediction), 
                     # #tf.reduce_max(prediction), 
                     # #probabilities, 
                     # #cross_entropy,
                     # weights['h1'],
                     # biases['b1'],
                     # X], 
               # message="MAE info: ", summarize=20)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

test_data = {X: X_test, y: y_test}

for i in range(num_epochs):
    idxs = np.random.choice(train_N, size=train_N, replace=False)
    X_train_batch = X_train[idxs]
    y_train_batch = y_train[idxs]
    train_data = {X: X_train_batch, y: y_train_batch}
    
    # train step
    #sess.run(cross_entropy_print, feed_dict=train_data)
    sess.run(train_step, feed_dict=train_data)
    
    if i % print_every == 0:
      # find train and test MAE of current model
      train_acc, train_MAE = sess.run([accuracy, MAE], feed_dict=train_data)
      test_acc, test_MAE = sess.run([accuracy, MAE], feed_dict=test_data)
      print("Train MAE %.5f, train acc: %.5f, test MAE %.5f, test acc: %.5f" % (train_MAE, train_acc, test_MAE, test_acc))