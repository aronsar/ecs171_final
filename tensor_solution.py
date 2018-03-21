import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tutorials/data/MNIST/", one_hot=True)
print("Extraction of images is complete.")

data = np.load('train_data_sanitized.npy')
train_N = int(data.shape[0] * .8)
test_N = data.shape[0] - train_N
dim = data.shape[1] - 1 # the -1 is b/c the last element is the label (ie loss)
numEpochs = 500
batch_size = 50
#mu = .9 # referred to as momentum, but basically equivalent to friction coefficient
learning_rate = .0000000002
n_hidden1 = 150
n_hidden2 = 50

X_train = data[:train_N, 0:-1]
y_train = data[:train_N, -1] * 100 # the *100 is b/c data_sanitizer normalized the loss
X_test  = data[train_N:, 0:-1]
y_test  = data[train_N:, -1] * 100

X = tf.placeholder(tf.float32, [None, dim])
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

hidden1_out = relu(add(mult(X, W1), b1))
hidden2_out = relu(add(mult(hidden1_out, W2), b2))
prediction = add(mult(hidden2_out, Wout), bout)

train_MAE = np.sum(np.abs(y_train_batch - prediction.astype(int))) / batch_size

# we left off here -- we need to finish the computational graph stuff
# and then also do the session running stuff
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()

Y = tf.nn.softmax(tf.matmul(X, W) + b)

Y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.arg_max(Y_, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)


# no. of iterations
n_iter = 2000

# test set
test_data = {X: mnist.test.images, Y_: mnist.test.labels}

# lists to hold train accuracy and cross-entropy
acc_train_li = []
cross_train_li = []

# lists to hold test accuracy and cross-entropy
acc_test_li = []
cross_test_li = []

for i in range(n_iter):
    # load batch of images and correct answer
    bacth_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: bacth_X, Y_: batch_Y}
    
    # train
    sess.run(train_step, feed_dict=train_data)
    
    # find accuracy and cross entropy on current data
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    acc_train_li.append(a)
    cross_train_li.append(c)
    
    # find accuracy and cross entropy on test data
    a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    acc_test_li.append(a)
    cross_test_li.append(c)
    
print('Train Set Accuracy: {} \t Train Set cross-entropy Loss: {}'.format(acc_train_li[-1], cross_train_li[-1]))
print('Test Set Accuracy: {} \t Test Set cross-entropy Loss: {}'.format(acc_test_li[-1], cross_test_li[-1]))