"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
from common import DataOwner
import time
#from common import LogisticRegression
import tensorflow as tf
from tf_encrypted.protocol.aby3.model.logistic_regression import LogisticRegression
from common import ModelOwner
from tf_encrypted.protocol.aby3 import ABY3
import numpy as np


#@tfe.protocol.ABY3.define_local_computation('server0')
def provide_input():
    # normal TensorFlow operations can be run locally
    # as part of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(5, 10))
#@tfe.protocol.ABY3.define_local_computation('server1')
def provide_input1():
    # normal TensorFlow operations can be run locally
    # as part of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(10, 10))
def main(servers):
    num_features = 20
    training_set_size = 2000
    test_set_size = 100
    batch_size = 100
    num_batches = (training_set_size // batch_size) * 10

    model_owner = ModelOwner("server0")
    data_owner_0 = DataOwner(
    "server0", num_features, training_set_size, test_set_size, batch_size // 2
    )
    data_owner_1 = DataOwner(
    "server1", num_features, training_set_size, test_set_size, batch_size // 2
    )
    tfe.set_protocol(ABY3(tfe.get_config().get_player(data_owner_0.player_name), tfe.get_config().get_player(data_owner_1.player_name), tfe.get_config().get_player("server2")))

    x_train_0 = tfe.define_private_input(
            data_owner_0.player_name,
            provide_input
              )
    x_train_1 = tfe.define_private_input(
            data_owner_1.player_name,
            provide_input1
              )
    
    x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))
    y = tfe.define_private_variable(tf.constant([[7, 8], [9, 10], [11, 12]]))

    
  
   # y_train = tfe.gather(x_train_1, 0, axis=1)
#     y_train = x_train_1[:,1]

#     #Remove bob's first column (which is label)
#    # x_train_1 = tfe.strided_slice(x_train_1, [0,1], [x_train_1.shape[0],x_train_1.shape[1]], [1,1]) 
  
#     x_train = tfe.concat([x_train_0, x_train_1], axis=0)

#     # x = tfe.define_private_variable(tf.ones(shape=(2, 2)) )
#     # y = tfe.define_private_variable(tf.ones(shape=(2, 1)))

#     x_train = tfe.concat([x_train_0, x_train_1], axis=0)
#     y_train = tfe.concat([y_train, y_train], axis=0)

#     model = LogisticRegression(num_features)
#     reveal_weights_op = model_owner.receive_weights(model.weights)
    
    with tfe.Session() as sess:
        sess.run(tfe.global_variables_initializer(), tag = "init")
        #y = tfe.matmul(x_train_0, x_train_1)
        z = tfe.matmul(x, y)
        T1 = time.clock()
        result = sess.run(z.reveal())
        T2 = time.clock()
        print((T2-T1)*1000)
        print(result)
       # model.fit(sess, x_train, y_train, num_batches)
        # TODO(Morten)
        # each evaluation results in nodes for a forward pass being added to the graph;
        # maybe there's some way to avoid this, even if it means only if the shapes match
        #model.evaluate(sess, x_test_0, y_test_0, data_owner_0)
        #model.evaluate(sess, x_test_1, y_test_1, data_owner_1)
    
        # TODO(Morten)
        # each evaluation results in nodes for a forward pass being added to the graph;
        # maybe there's some way to avoid this, even if it means only if the shapes match
    #   model.evaluate(sess, x_test_0, y_test_0, data_owner_0)
        #   model.evaluate(sess, x_test_1, y_test_1, data_owner_1)

       # sess.run(reveal_weights_op, tag="reveal")



def start_master(cluster_config_file=None):
  print("Starting alice...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  tfe.set_protocol(tfe.protocol.ABY3(remote_config.players[0].name,remote_config.players[1].name,remote_config.players[2].name))
  players = remote_config.players
  server0 = remote_config.server(players[0].name)
  main(server0)


if __name__ == "__main__":
  start_master("config.json")