"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import tensorflow as tf
import numpy as np
import time
from common import DataOwner, ModelOwner, DataSchema
from tf_encrypted.protocol.aby3 import ARITHMETIC
from tf_encrypted.protocol.aby3 import BOOLEAN
from tf_encrypted.protocol.aby3.model.logistic_regression import LogisticRegression
def main(server):
        num_features = 32
        batch_size = 100
 

  #who shall receive the output
        model_owner = ModelOwner('alice')
    
        data_schema0 = DataSchema([tf.float64]*16, [0.0]*16) 
        data_schema1 = DataSchema([tf.int64]+[tf.float64]*16, [0]+[0.0]*16) 
        data_owner_0 = DataOwner('alice',
            'aliceTrainFile.csv',
            data_schema0,
            batch_size = batch_size,
            num_features = num_features)
        data_owner_1 = DataOwner('bob',
            'bobTrainFileWithLabel.csv',
            data_schema1,
            batch_size = batch_size,
            num_features = num_features)
        data_owner_2 = DataOwner('crypto-producer',
            'bobTrainFileWithLabel.csv',
            data_schema1,
            batch_size = batch_size,
            num_features = num_features)
        prot = tfe.protocol.ABY3(
        tfe.get_config().get_player(data_owner_0.player_name),
        tfe.get_config().get_player(data_owner_1.player_name), "crypto-producer")
        tfe.set_protocol(prot)

        x1 = tfe.define_private_input( data_owner_0.player_name, data_owner_0.random_vector)
        x2 = tfe.define_private_input( data_owner_1.player_name, data_owner_1.random_vector)
        tmppp = np.logspace(-2, 3, 1000)
        input =  tfe.define_private_variable(tmppp)
        
        res111 = tfe.inverse_sqrt(input)
        
        truth = np.power(tmppp, -0.5)

        with tfe.Session() as sess:
            sess.run(tfe.global_variables_initializer(),
                    tag='init')
            xxx = []
            #tmp = sess.run(res111.reveal(), tag='reveal')
            T1 = time.time()
            times = []
            for i in range(0, 1):
                tmp = sess.run(res111.reveal(), tag='reveal')
            T2 = time.time()
            print(tmp - truth)
            print(np.linalg.norm(tmp - truth, ord = 1))
            np.save('test_tf_error.npy', tmp - truth)
            print(((T2-T1)))
    
  
def start_master(cluster_config_file=None):
  print("Starting alice...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  players = remote_config.players
  server0 = remote_config.server(players[0].name)
  main(server0)


if __name__ == "__main__":
  start_master("config.json")