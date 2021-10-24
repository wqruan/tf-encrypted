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

        x = x1+x2
        res111 = tfe.inverse_sqrt(x)
    #     z = tfe.A2B(x)
    #     results = []
    #     for i in range(0, prot.nbits):
    #         results.append(tfe.bit_extract(z, i))

    #     for i in range(2, prot.nbits):
    #         results[prot.nbits - i - 1] = prot.B_or(results[prot.nbits - i - 1], results[prot.nbits - i])

    #     is_odd = prot.B_xor(results[prot.nbits - 2], results[prot.nbits - 3])

    #     for i in range(4, prot.nbits+1):
    #         is_odd = prot.B_xor(is_odd, results[prot.nbits - i])

    #     tmp1 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC, apply_scaling = False)
    #     tmp2 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC)
    #     is_odd = tfe.mul_AB(tmp2, is_odd)
    #     #is_odd = tfe.share_conversion_b_a(is_odd)

    #     #results = tfe.shares_conversion_b_a(results);
    #     exp = tfe.mul_AB(tmp1, results[0])
    #    # exp = results[0]
    #     b = (1 - exp)*(2**(prot.nbits-2)) + 1
    #     exp = exp*tmp2
    #     for i in range(1, prot.nbits-1):
    #         tmp00 = tfe.mul_AB(tmp1, results[i])
    #         #tmp00 = results[i]#tfe.share_conversion_b_a(results[i])
    #         exp += tmp00*tmp2
    #         b += (1-tmp00)*(2**(prot.nbits-2-i))
    #     b = tfe.truncate(b)
    #     b = tfe.truncate(b)
    #     b = tfe.truncate(x*b*4)
    #     b = tfe.truncate(b)

    #     b = tfe.polynomial(b, [2.223, -2.046, 0.8277])
    #     # exp = exp - prot.fixedpoint_config.precision_fractional
    #     exp = prot.fixedpoint_config.precision_fractional - ((exp - prot.fixedpoint_config.precision_fractional)*0.5)

    #     exp_b = tfe.A2B(exp)

    #     exp_bs = []
    #     for i in range(prot.fixedpoint_config.precision_fractional, prot.fixedpoint_config.precision_fractional+5):
    #         exp_bs.append(tfe.bit_extract(exp_b, i))
        
    #     bs = []
    #     for i in range(0, len(exp_bs)):
    #         bs.append(tfe.mul_AB(tmp2, exp_bs[i]))

        
    #     ibs = []
    #     for i in range(0, len(exp_bs)):
    #         ibs.append(1 - bs[i])

    #     exp_sqrt = ((2**1) * bs[0] + ibs[0]);
    #     for i in range(1, len(bs)):
    #         exp_sqrt = exp_sqrt * ((2**(2**i)) * bs[i] + ibs[i])
        
    #     exp_sqrt = tfe.truncate(exp_sqrt)

    #     exp_sqrt_odd = exp_sqrt * (2**(0.5))

    #     exp_sqrt = exp_sqrt *  (1 - is_odd) + exp_sqrt_odd * (is_odd)
    #     assert z.share_type == BOOLEAN
       
    #     res111 = b * exp_sqrt

        with tfe.Session() as sess:
            sess.run(tfe.global_variables_initializer(),
                    tag='init')
            xxx = []
            #tmp = sess.run(res111.reveal(), tag='reveal')
            T1 = time.time()
            times = []
            for i in range(0, 100):
                tmp = sess.run(res111.reveal(), tag='reveal')
            T2 = time.time()
            
            print(((T2-T1)))
            print(tmp)
            
    
  
def start_master(cluster_config_file=None):
  print("Starting alice...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  players = remote_config.players
  server0 = remote_config.server(players[0].name)
  main(server0)


if __name__ == "__main__":
  start_master("config.json")