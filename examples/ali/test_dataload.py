"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import tensorflow as tf
from common import DataOwner, ModelOwner, DataSchema,LogisticRegression_new
from tf_encrypted.protocol.aby3.model.logistic_regression import LogisticRegression
import numpy as np

def main(server):
  #imdb : 0.86
  
  #cifar : 0.81
  
  #mnist : 0.8
  #  0.80632, 0.80712, 0.81288, 0.80364, 0.79776
  #imdb : 0.84
  #mnist :0.77
  #cifar : 0.782
    num_rows = 5000
    num_features = 768
    num_epoch = 1
    batch_size = 128
    num_batches = 391 #(num_rows // batch_size ) * num_epoch
    noise_multiplier = 0.86 # nonInitialization:0.78 imdb:1.206 cifar:1.14
    C = 3
    class_num = 1
    #who shall receive the output
    model_owner = ModelOwner('alice')
    
    data_schema0 = DataSchema([tf.int64]+[tf.float64]*num_features, [0]+ [0.0]*num_features) 
    data_schema1 = DataSchema([tf.int64]+[tf.float64]*num_features, [0]+ [0.0]*num_features) 
    data_owner_0 = DataOwner('alice',
            '',
            data_schema0,
            train_file = '/disk/wqruan/tf-encrypted/examples/ali/mnist/imdb_trainset.csv',
            #  train_label_file = '/disk/wqruan/SMMLF/features/cifar-10_train_label.csv',
            num_features = num_features,
            class_num = class_num,
            batch_size= batch_size,
            C=C,
            noise_multiplier= noise_multiplier)
    data_owner_1 = DataOwner('bob',
            '',
            data_schema1,
            test_file = '/disk/wqruan/tf-encrypted/examples/ali/mnist/imdb_testset.csv',
            # test_label_file = '/disk/wqruan/SMMLF/features/cifar-10_test_label.csv',
            num_features = num_features,
            class_num = class_num,
             batch_size= batch_size,
             C=C,
            noise_multiplier= noise_multiplier)
    data_owner_2 = DataOwner('crypto-producer',
         '',
         data_schema1,
         batch_size = batch_size,
         num_features = num_features,
         class_num = class_num,
         C = C,
         noise_multiplier= noise_multiplier)
    tfe.set_protocol(tfe.protocol.ABY3(
        tfe.get_config().get_player(data_owner_0.player_name),
        tfe.get_config().get_player(data_owner_1.player_name), "crypto-producer", use_noninteractive_truncation=True))

    train_data, train_label = tfe.define_private_input(
                data_owner_0.player_name,
                data_owner_0.provide_train_data
                )
    test_data,test_label = tfe.define_private_input(
                data_owner_1.player_name,
                data_owner_1.provide_test_data
                )
    noise_0 = tfe.define_private_input( data_owner_0.player_name, data_owner_0.provide_noise)
    noise_1 = tfe.define_private_input( data_owner_1.player_name, data_owner_1.provide_noise)
    noise_2 = tfe.define_private_input( data_owner_2.player_name, data_owner_2.provide_noise)
    noise = noise_0 + noise_1 + noise_2

    model = LogisticRegression_new(num_features, class_num, batch_size)
#   reveal_weights_op = model_owner.receive_weights(model.weights)
    with tfe.Session() as sess:
        sess.run([tfe.global_variables_initializer(), data_owner_0.initializer], 
                tag='init')

        model.fit(sess, train_data, train_label,test_data, test_label, num_batches, noise,data_owner_0)
        # for i in range(0, 392):
        #   tmp = sess.run(train_label.reveal())
        #   print(np.sum(tmp[:, 0]))
          

        # # TODO(Morten)
        # # each evaluation results in nodes for a forward pass being added to the graph;
        # # maybe there's some way to avoid this, even if it means only if the shapes match
        # model.evaluate(sess, x_train, y_train, data_owner_0)
        # model.evaluate(sess, test_data, test_label, data_owner_0)
        # for i in range(10):
        #     result = sess.run(test_data[i].reveal())
        #     print(result) 
        # sess.run(reveal_weights_op, tag='reveal')
  
  
def start_master(cluster_config_file=None):
  print("Starting alice...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  players = remote_config.players
  server0 = remote_config.server(players[0].name)
  main(server0)


if __name__ == "__main__":
  start_master("config.json")