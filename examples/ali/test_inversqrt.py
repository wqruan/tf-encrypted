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
        
        # res111 = tfe.inverse_sqrt(x1)
        
        tmp = np.zeros([1280, 1280])
        for i in range(len(tmp)):
            indice = (i%10)*128+ int(i/10)
            # tmp[i, indice] = 1
            tmp[indice, i ] = 1
        tmp1 = np.zeros([1280, 3])
        for i in range(len(tmp1)):
            tmp1[i, 0] = i
            tmp1[i, 1] = i+1
            tmp1[i, 2] = i+2
        tmp1 = np.transpose(tmp1)
        tmp2 = tmp1.dot(tmp)
        for i in range(20):
            print(tmp2[:,i])

        # with tfe.Session() as sess:
        #     sess.run(tfe.global_variables_initializer(),
        #             tag='init')
        #     xxx = []
        #     #tmp = sess.run(res111.reveal(), tag='reveal')
        #     T1 = time.time()
        #     times = []
        #     for i in range(0, 100):
        #         tmp = sess.run(res111.reveal(), tag='reveal')
        #     T2 = time.time()
            
        #     print(((T2-T1)))
        #     print(tmp)
            
    
  
def start_master(cluster_config_file=None):
  print("Starting alice...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  players = remote_config.players
  server0 = remote_config.server(players[0].name)
  main(server0)


if __name__ == "__main__":
  start_master("config.json")