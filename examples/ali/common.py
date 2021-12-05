"""Provide classes to perform private training and private prediction with
logistic regression"""
import tensorflow as tf
import tf_encrypted as tfe
import math
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
# class LogisticRegression:
#   """Contains methods to build and train logistic regression."""
#   def __init__(self, num_features):
#     self.w = tfe.define_private_variable(
#         tf.random_uniform([num_features, 1], -0.01, 0.01))
#     self.w_masked = tfe.mask(self.w)
#     self.b = tfe.define_private_variable(tf.zeros([1]))
#     self.b_masked = tfe.mask(self.b)

#   @property
#   def weights(self):
#     return self.w, self.b

#   def forward(self, x):
#     with tf.name_scope("forward"):
#       out = tfe.matmul(x, self.w) + self.b
#       y = tfe.sigmoid(out)
#       return y

#   def backward(self, x, dy, learning_rate=0.01):
#     batch_size = x.shape.as_list()[0]
#     with tf.name_scope("backward"):
#       # store = []
#       # for i in range(10):
#       #       store.append(tfe.diag(z[:,i]))
#       # tmppp = tfe.concat(store, axis = 0)
#       # gradients = tfe.matmul(tmppp, x)
#       # gradients = tfe.reshape(gradients, [128, 40960])
#       # gradients = gradients*gradients
#       # norm_square = tfe.reduce_sum(gradients, axis=1)
#       # norm_inverse = tfe.inverse_sqrt(norm_square)
#       # C = 5
#       # norm_inverse = norm_inverse * C
#       # z1 = tfe.polynomial_piecewise(
#       #       norm_inverse,
#       #       (0, 1),
#       #       ((0,), (0, 1), (1,)),  # use tuple because list is not hashable
#       # )
#       # z1 = tfe.reshape(z1,[1,128])
#       # gradients_clipped = tfe.matmul(z1, gradients)
#       # gradients_clipped = tfe.reshape(gradients_clipped, [10, 4096])
#       dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
#       db = tfe.reduce_sum(dy, axis=0) / batch_size
#       dw_norm_inverse = tfe.inverse_sqrt(x)
#       assign_ops = [
#           tfe.assign(self.w, self.w - dw * learning_rate),
#           tfe.assign(self.b, self.b - db * learning_rate),
#       ]
#       return assign_ops

#   def loss_grad(self, y, y_hat):
#     with tf.name_scope("loss-grad"):
#       dy = y_hat - y
#       return dy

#   def fit_batch(self, x, y):
#     with tf.name_scope("fit-batch"):
#       y_hat = self.forward(x)
#       dy = self.loss_grad(y, y_hat)
#       fit_batch_op = self.backward(x, dy)
#       return fit_batch_op

#   def fit(self, sess, x, y, num_batches):
#     fit_batch_op = self.fit_batch(x, y)
#     for batch in range(num_batches):
#       print("Batch {0: >4d}".format(batch))
#       sess.run(fit_batch_op, tag='fit-batch')

#   def evaluate(self, sess, x, y, data_owner):
#     """Return the accuracy"""
#     def print_accuracy(y_hat, y) -> tf.Operation:
#       with tf.name_scope("print-accuracy"):
#         correct_prediction = tf.equal(tf.round(y_hat), y)
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         print_op = tf.print("Accuracy on {}:".format(data_owner.player_name),
#                             accuracy)
#         return print_op

#     with tf.name_scope("evaluate"):
#       y_hat = self.forward(x)
#       print_accuracy_op = tfe.define_output(data_owner.player_name,
#                                             [y_hat, y],
#                                             print_accuracy)

#     sess.run(print_accuracy_op, tag='evaluate')

class LogisticRegression_new:
  """Contains methods to build and train logistic regression."""
  def __init__(self, num_features, class_num, batch_size):
      # print(num_features)
      initial_model = np.loadtxt("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/models/imdbinitial_model.csv",delimiter = ",")
      self.w = tfe.define_private_variable(np.reshape(initial_model.T, (num_features + 1, class_num)))
      # self.w = tfe.define_private_variable(tf.zeros([10,10]))
      # self.w = tfe.define_private_variable(tf.zeros([num_features + 1, class_num]))
      # self.w = tfe.define_private_variable(tf.random_normal([num_features + 1, class_num], 0, 0.2))
      self.class_num = class_num
      self.num_features = num_features
      self.correct =0.0
      self.batch_size = batch_size
      self.record = []
  @property
  def weights(self):
    return self.w

  def forward(self, x):
    with tf.name_scope("forward"):
      out = tfe.matmul(x, self.w)
      y = tfe.sigmoid(out)
      return y

  def backward(self,sess, x, dy, noise,learning_rate=0.2):
    batch_size = x.shape.as_list()[0]
    with tf.name_scope("backward"):
      store = []

      if self.class_num >2 :
        for i in range(0, self.class_num):
              store.append(tfe.diag(dy[:,i]))
        tmppp = tfe.concat(store, axis = 1)
        gradients = tfe.matmul( tfe.transpose(x), tmppp)
        permutation = np.zeros([batch_size*10, batch_size*10])
        for i in range(len(permutation)):
              indice = (i%10)*batch_size+ int(i/10)
              permutation[indice, i] = 1
        permutation = tfe.define_constant(permutation)
        gradients = gradients.matmul(permutation)
        gradients = tfe.transpose(gradients)
        gradients = tfe.reshape(gradients, [batch_size, (self.num_features+1)*self.class_num])
      else:
        gradients = x * dy
      gradients_square = gradients*gradients
      norm_square = tfe.reduce_sum(gradients_square, axis=1)
      norm_inverse = tfe.inverse_sqrt(norm_square)
      C = 3
      norm_inverse = norm_inverse * C
      z1 = tfe.polynomial_piecewise(
            norm_inverse,
            (0, 1),
            ((0,), (0, 1), (1,)), 
      )
      z1 = tfe.reshape(z1,[1,batch_size])
      gradients_clipped = tfe.matmul(z1, gradients)
      gradients_clipped = tfe.reshape(gradients_clipped, [self.class_num, self.num_features+1])
      gradients_clipped = gradients_clipped + noise
      gradients_clipped = gradients_clipped / batch_size
      gradients_clipped =  tfe.transpose(gradients_clipped)
      assign_ops = [
          tfe.assign(self.w, self.w - gradients_clipped * learning_rate),
      ]
      # self.dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
      # db = tfe.reduce_sum(dy, axis=0) / batch_size
      # dw_norm_inverse = tfe.inverse_sqrt(x)
      # assign_ops = [
      #     tfe.assign(self.w, self.w - self.dw * learning_rate),
      # ]
      
      return assign_ops

  def loss_grad(self, y, y_hat):
    with tf.name_scope("loss-grad"):
      if self.class_num == 1:
        y = tfe.reshape(y, [self.batch_size, 1])
      dy = y_hat - y
      return dy

  def fit_batch(self, sess, x, y, noise):
    with tf.name_scope("fit-batch"):
      y_hat = self.forward(x)
      dy = self.loss_grad(y, y_hat)
      fit_batch_op = self.backward(sess, x, dy, noise)
      return fit_batch_op

  def test(self, X_test,Y_test,theta, class_num):
      tmp = X_test.dot(theta.T)
      tmp = 1 / (1 + np.exp(-tmp))
      correct = 0;
      if class_num > 2:
          for i in range(len(tmp)):
              res = np.argmax(tmp[i])
              if(Y_test[i][res]>0):
                  correct+=1;
      else:
          for i in range(len(tmp)):
              if(abs((Y_test[i] - tmp[i]))<0.5):
                  correct+=1;
      print(correct)
      return  correct/len(Y_test)       

        
  def fit(self, sess, x, y, x_test , y_test,num_batches, noise, data_owner):
    # X_test  = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/transfer/features/simclr_r50_2x_sk1_cifar_test.npy")
    # Y_test  =np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/transfer/features/cifar-test-y.npy")
    # # X_test  = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/raw_data/cifar-testraw-x.npy")
    # Y_test  =np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/raw_data/cifar-testraw-y.npy")
    # X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis = 1)
    # Y_test = np.eye(10)[Y_test.astype(int)]
    # X_test  = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/features/mnist_test_hog_x.npy")
    # Y_test  =np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/features/mnist_test_hog_y.npy")
    # X_test  = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/raw_data/mnist_test_x.npy")
    # Y_test  = np.load("//disk/wqruan/Pretrain/Handcrafted-DP/transfer/raw_data/mnist_test_y.npy")        
    # Y_test = np.eye(10)[Y_test.astype(int)]    
    # X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis = 1)
    X_test = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/features/imdb_test_x.npy")
    Y_test = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/features/imdb_test_y.npy")
    # test = np.load("/disk/wqruan/Pretrain/Handcrafted-DP/transfer/raw_data/imdb_test.npy")
    # X_test= test[:, 1:]
    # X_test = X_test /15000
    # Y_test = test[:, 0]
    Y_test = Y_test.reshape((len(Y_test), 1))
    X_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis = 1)
    
    file = open("imdb-test.txt", 'a+')
    www = sess.run(self.w.reveal())
    print(www)
    self.test(X_test, Y_test, www.T, self.class_num)   
    fit_batch_op = self.fit_batch(sess, x, y, noise)
    i=0
    for batch in range(num_batches):
      if batch>=10 and batch%int(math.pow(10,int(math.log10(batch))))== 0:
          www = sess.run(self.w.reveal())
          tmp = self.test(X_test, Y_test, www.T, self.class_num)
          file.write("iteration num: " + str(batch))
          file.write("accuracy: " + str(tmp))
          file.write("\n")

      print("Batch {0: >4d}".format(batch))
      sess.run(fit_batch_op, tag='fit-batch')

      file.flush()
    www = sess.run(self.w.reveal())
    self.test(X_test, Y_test, www.T, self.class_num)   
    tmp = self.test(X_test, Y_test, www.T, self.class_num)
    file.write("iteration num: " + str(num_batches))
    file.write("accuracy: " + str(tmp))
    file.write("\n")

          
 

  def evaluate(self, sess, x, y, batch,data_owner):
    """Return the accuracy"""
    def print_accuracy(y_hat, y) -> tf.Operation:  
      with tf.name_scope("print-accuracy"):
        correct = 0.0
        res = tf.argmax(y_hat, 1)
        res_1 = tf.argmax(y, 1)
        correct_prediction = tf.equal(res, res_1)
        assign_ops = [
          tf.assign(self.correct, self.correct + tf.reduce_sum(correct_prediction)),
        ] 
        sess.run(assign_ops)
        tmp = tf.print("correct:",tf.reduce_sum(correct_prediction))
        sess.run(tmp)
        tmp1 = tf.print("correct:",self.correct)
        sess.run(tmp1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print_op = tf.print("")
 
        return print_op

    with tf.name_scope("evaluate"):
      
      y_hat = self.forward(x)
      for i in range(0, 50):
        print_accuracy_op = tfe.define_output(data_owner.player_name,
                                            [y_hat, y],
                                            print_accuracy)
        sess.run(print_accuracy_op, tag='evaluate')
      tmp = self.correct/(50*x.shape.as_list()[0])
      tmp = sess.run(tmp)
      print(tmp)
      self.record.append([batch, tmp])

class DataOwner:
  """Contains code meant to be executed by a data owner Player."""
  def __init__(self, player_name, local_data_file, data_schema,
         C = 1, noise_multiplier = 0.92,  train_file = '', train_label_file = '', test_file = '', test_label_file ='', class_num = 1,header=False, index=False, field_delim=',', na_values=['nan'], batch_size=128, num_features = 32, mu = 0, sigma = 0.001):
    self.player_name = player_name
    self.local_data_file = local_data_file
    self.data_schema = data_schema
    self.batch_size = batch_size
    self.header = header
    self.index = index
    self.na_values = na_values
    self.field_delim = field_delim
    self.num_features = num_features
    self.mu = mu
    self.sigma = sigma
    self.train_file = train_file
    self.train_label_file = train_label_file
    self.test_file = test_file
    self.class_num = class_num
    self.test_label_file = test_label_file
    self.C = C
    self.noise_multiplier = noise_multiplier
    self.train_initializer = None
    tmp = list(player_name)
    self.ran = 0
    for i in range(0, len(tmp)):
        self.ran += ord(tmp[i])

  @property
  def initializer(self):
    return self.train_initializer

  def provide_data(self):

    def decode(line):
      fields = tf.string_split([line], self.field_delim).values
      if self.index: # Skip index
        fields = fields[1:]
      fields = tf.regex_replace(fields, '|'.join(self.na_values), 'nan')
      fields = tf.string_to_number(fields, tf.float32)
      return fields

    def fill_na(fields, fill_values):
      fields = tf.where(tf.is_nan(fields), fill_values, fields)
      return fields

    dataset = tf.data.TextLineDataset(self.local_data_file)
    if self.header: # Skip header
      dataset = dataset.skip(1)
    dataset = dataset\
        .map(decode)\
        .map(lambda x: fill_na(x, self.data_schema.field_defaults))\
        .repeat()\
        .shuffle(buffer_size=10000)\
        .batch(self.batch_size)
    iterator = dataset.make_one_shot_iterator()
    self.train_initializer = iterator.initializer
    batch = iterator.get_next()
    batch = tf.reshape(batch, [self.batch_size, self.data_schema.field_num])
    return batch
  
  def provide_train_data(self):
    def decode(line):
      fields = tf.string_split([line], self.field_delim).values
      if self.index: # Skip index
          fields = fields[1:]
      fields = tf.regex_replace(fields, '|'.join(self.na_values), 'nan')
      fields = tf.string_to_number(fields, tf.float32)
      return fields

    def fill_na(fields, fill_values):
      fields = tf.where(tf.is_nan(fields), fill_values, fields)
      return fields

    dataset = tf.data.TextLineDataset(self.train_file)
    if self.header: # Skip header
      dataset = dataset.skip(1)
    dataset = dataset\
        .map(decode)\
        .map(lambda x: fill_na(x, self.data_schema.field_defaults))\
        .repeat()\
        .shuffle(buffer_size=10000)\
        .batch(self.batch_size)
    iterator = dataset.make_initializable_iterator()
    self.train_initializer = iterator.initializer
    batch = iterator.get_next()
    batch = tf.reshape(batch, [self.batch_size, self.data_schema.field_num])
    if self.train_label_file == '':
      train_label = batch[:, 0]
      if self.class_num > 2:
        train_label = tf.one_hot(tf.cast(train_label, dtype = tf.int32), self.class_num)
      train_data = batch[:, 1:]  
      # train_data = train_data /15000
      # train_data = 20 * train_data / tf.norm(train_data, ord = 2)
      bias_term = tf.ones([self.batch_size, 1])
        
      return tf.concat([train_data,bias_term], axis = 1), train_label
    batch = tf.reshape(batch, [self.batch_size, self.num_features])
    labels = tf.data.TextLineDataset(self.train_label_file)
    if self.header: # Skip header
      labels = labels.skip(1)
    labels = labels\
        .map(decode)\
        .repeat()\
        .batch(self.batch_size)
    iterator1 = labels.make_one_shot_iterator()
    batch_labels = iterator1.get_next()
    batch_labels = tf.reshape(batch_labels, [self.batch_size])
    batch_labels = tf.one_hot(tf.cast(batch_labels, dtype = tf.int32), self.class_num)
    bias_term = tf.ones([self.batch_size, 1])
    return tf.concat([batch, bias_term], axis = 1), batch_labels


  def provide_test_data(self):
    def decode(line):
      fields = tf.string_split([line], self.field_delim).values
      if self.index: # Skip index
          fields = fields[1:]
      fields = tf.regex_replace(fields, '|'.join(self.na_values), 'nan')
      fields = tf.string_to_number(fields, tf.float32)
      return fields

    def fill_na(fields, fill_values):
      fields = tf.where(tf.is_nan(fields), fill_values, fields)
      return fields

    dataset = tf.data.TextLineDataset(self.test_file)
    if self.header: # Skip header
      dataset = dataset.skip(1)
    dataset = dataset\
        .map(decode)\
        .map(lambda x: fill_na(x, self.data_schema.field_defaults))\
        .repeat()\
        .batch(self.batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    batch = tf.reshape(batch, [self.batch_size, self.data_schema.field_num])
    if self.test_label_file == '':
 
      test_label = batch[:, 0]
      test_label = tf.one_hot(tf.cast(test_label, dtype = tf.int32), self.class_num)
      test_data = batch[:, 1:]    
      # test_data = 20 * test_data / tf.norm(test_data, ord = 2)  
      bias_term = tf.ones([self.batch_size, 1])
      return tf.concat([test_data, bias_term], axis = 1), test_label
    batch = tf.reshape(batch, [self.batch_size, self.num_features])
    labels = tf.data.TextLineDataset(self.test_label_file)
    if self.header: # Skip header
      labels = labels.skip(1)
    labels = labels\
        .map(decode)\
        .repeat()\
        .batch(self.batch_size)
    iterator1 = labels.make_one_shot_iterator()
    batch_labels = iterator1.get_next()
    batch_labels = tf.reshape(batch_labels, [self.batch_size])
    batch_labels = tf.one_hot(tf.cast(batch_labels, dtype = tf.int32), self.class_num)
    bias_term = tf.ones([self.batch_size, 1])
    return tf.concat([batch, bias_term], axis = 1), batch_labels

  def provide_noise(self):
        
    local_noise = tf.random_normal([self.class_num,self.num_features+1], mean = self.mu, stddev = (1.5**0.5)*self.noise_multiplier*self.C/(3**0.5), seed = time.clock() - self.ran)
    return local_noise;

  def random_vector(self):
    random_vector = tf.ones([64])
    return random_vector
class DataSchema:
  def __init__(self, field_types, field_defaults):
    self.field_types = field_types
    self.field_defaults = field_defaults

  @property
  def field_num(self):
    return len(self.field_types)

class ModelOwner:
  """Contains code meant to be executed by a model owner Player."""
  def __init__(self, player_name):
    self.player_name = player_name

  @tfe.local_computation
  def receive_weights(self, *weights):
    return tf.print("Weights on {}:".format(self.player_name), weights)


class PredictionClient:
  """Contains methods meant to be executed by a prediction client."""
  def __init__(self, player_name, num_features):
    self.player_name = player_name
    self.num_features = num_features

  @tfe.local_computation
  def provide_input(self):
    return tf.random.uniform(
        minval=-.5,
        maxval=.5,
        dtype=tf.float32,
        shape=[1, self.num_features])

  @tfe.local_computation
  def receive_output(self, result):
    return tf.print("Result on {}:".format(self.player_name), result)
