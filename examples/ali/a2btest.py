import os
import tempfile
import unittest
import time
import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.operations.secure_random import secure_random
from tf_encrypted.protocol.aby3 import ABY3
from tf_encrypted.protocol.aby3 import ARITHMETIC
from tf_encrypted.protocol.aby3 import BOOLEAN
from typing import Tuple

def test_mat():
        tf.reset_default_graph()

        prot = ABY3(use_noninteractive_truncation=True)
        tfe.set_protocol(prot)
        x = tfe.define_private_variable(
            tf.constant(np.ones((128, 4096))), share_type=ARITHMETIC
        )
        y = tfe.define_private_variable(
            tf.constant(4), share_type=ARITHMETIC
        )
        z = tfe.define_private_variable(
            tf.constant(np.ones((128, 10))), share_type=ARITHMETIC
        )
        # z = tfe.reshape(x, [2,4])
        store = []
        for i in range(10):
            store.append(tfe.diag(z[:,i]))
        tmppp = tfe.concat(store, axis = 0)
        gradients = tfe.matmul(tmppp, x)
        print(np.shape(gradients))
        gradients = tfe.reshape(gradients, [128, 40960])
        gradients = gradients*gradients
        norm_square = tfe.reduce_sum(gradients, axis=1)
        norm_inverse = tfe.inverse_sqrt(norm_square)
        C = 5
        norm_inverse = norm_inverse * C
        z1 = tfe.polynomial_piecewise(
            norm_inverse,
            (0, 1),
            ((0,), (0, 1), (1,)),  # use tuple because list is not hashable
        )
        z1 = tfe.reshape(z1,[1,128])
        gradients_clipped = tfe.matmul(z1, gradients)
        gradients_clipped = tfe.reshape(gradients_clipped, [10, 4096])
        # for i in range(10):
        #     tmp = z[:,i]
        #     tmp = tfe.reshape(tmp,[128,1])
        #     store.append(x*tmp)
        # for i in range(128):
        #     tmpstore = []
        #     for j in range(10):
        #         tmpstore.append(store[j][i])
        #     tmpstore = tfe.concat(tmpstore, axis = 0)
        #     gradients.append(tmpstore)
        # gradients = tfe.concat(gradients, axis = 1)
        #gradients = tfe.concat(gradients, axis = 1)
        # tmp2 = tfe.diag(z[0]);
        # tmp1 = tfe.diag(z[1]);
        # tmp = tfe.diag(z[2]);

        # tmp3 = tfe.concat([tmp,tmp1,tmp2], axis = 1)
        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            print("start")
            T1 = time.time()
            for i in range(10):
                 result = sess.run(gradients_clipped.reveal())
            T2 = time.time()
            print((T2-T1)*1000)
            print(result)
            print(np.shape(result))


def test_a2b_private():
        tf.reset_default_graph()

        prot = ABY3(use_noninteractive_truncation=True)
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant(np.logspace(-2, 3, 1000)), share_type=ARITHMETIC
        )
        ground_truth = 1/np.sqrt(np.logspace(-2,3,1000))
        z = tfe.A2B(x)
        results = []
        for i in range(0, prot.nbits):
            results.append(tfe.bit_extract(z, i))

        for i in range(2, prot.nbits):
            results[prot.nbits - i - 1] = prot.B_or(results[prot.nbits - i - 1], results[prot.nbits - i])

        # x_1 = []
        # x_2 = []
        # x_3 = []
        # x_4 = []
        # x_5 = []
        # for i in range(0,31):
        #     x_1.append(prot.B_or(results[62 - 2*i], results[62 - 2*i -1]))
        # for i in range(0, 15):
        #     x_2.append(prot.B_or(x_1[2 * i], x_1[2 * i+1 ]))
        # for i in range(0, 7):
        #     x_3.append(prot.B_or(x_2[2 * i], x_2[2 * i+1 ]))
        # for i in range(0, 3):
        #     x_4.append(prot.B_or(x_3[2 * i], x_3[2 * i+1 ]))
        # for i in range(0,1):
        #     x_5.append(prot.B_or(x_4[2 * i], x_4[2 * i+1 ]))
        # res = [None]*64;
        # res[63] = results[63]
        # res[62] = results[62]
        # res[61] = x_1[0]
        # res[59] = x_2[0]
        # res[55] = x_3[0]
        # res[47] = x_4[0]
        # res[31] = x_5[0]
        #     #first round
        # res[60] = prot.B_or(res[61], results[60])
        # res[58] = prot.B_or(res[59], results[58])
        # res[57] = prot.B_or(res[59], x_1[2])
        # res[54] = prot.B_or(res[55], results[54])
        # res[53] = prot.B_or(res[55], x_1[4])
        # res[51] = prot.B_or(res[55], x_2[2])
        # res[46] = prot.B_or(res[47], results[46])
        # res[45] = prot.B_or(res[47], x_1[8])
        # res[43] = prot.B_or(res[47], x_2[4])
        # res[39] = prot.B_or(res[47], x_3[2])
        # res[30] = prot.B_or(res[31], results[30])
        # res[29] = prot.B_or(res[31], x_1[16])
        # res[27] = prot.B_or(res[31], x_2[8])
        # res[23] = prot.B_or(res[31], x_3[4])
        # res[15] = prot.B_or(res[31], x_4[2])
        # #second round
        # res[56] = prot.B_or(res[57], results[56])
        # res[55] = prot.B_or(res[57], x_1[3])
        # res[52] = prot.B_or(res[53], results[52])
        # res[50] = prot.B_or(res[51], results[50])
        # res[49] = prot.B_or(res[51], x_1[6])
        # res[44] = prot.B_or(res[47], results[44])
        # res[42] = prot.B_or(res[43], results[42])
        # res[41] = prot.B_or(res[43], x_1[10])
        # res[38] = prot.B_or(res[39], results[38])
        # res[37] = prot.B_or(res[39], x_1[12])
        # res[35] = prot.B_or(res[39], x_2[6])
        # res[28] = prot.B_or(res[29], results[28])
        # res[26] = prot.B_or(res[27], results[26])
        # res[25] = prot.B_or(res[27], x_1[18])
        # res[22] = prot.B_or(res[23], results[22])
        # res[21] = prot.B_or(res[23], x_1[20])
        # res[19] = prot.B_or(res[23], x_2[10])
        # res[14] = prot.B_or(res[15], results[14])
        # res[13] = prot.B_or(res[15], x_1[24])
        # res[11] = prot.B_or(res[15], x_2[12])
        # res[7] = prot.B_or(res[15], x_3[6])
        # #third round
        # res[48] = prot.B_or(res[49], results[48])
        # res[40] = prot.B_or(res[41], results[40])
        # res[36] = prot.B_or(res[37], results[36])
        # res[34] = prot.B_or(res[35], results[34])
        # res[33] = prot.B_or(res[35], x_1[14])
        # res[24] = prot.B_or(res[25], results[24])
        # res[20] = prot.B_or(res[21], results[20])
        # res[18] = prot.B_or(res[19], results[18])
        # res[17] = prot.B_or(res[19], x_1[22])
        # res[12] = prot.B_or(res[13], results[12])
        # res[10] = prot.B_or(res[11], results[10])
        # res[9] = prot.B_or(res[11], x_1[26])
        # res[6] = prot.B_or(res[7], results[6])
        # res[5] = prot.B_or(res[7], x_1[28])
        # res[3] = prot.B_or(res[7], x_2[14])
        # #forth round
        # res[32] = prot.B_or(res[34], results[32])
        # res[16] = prot.B_or(res[17], results[16])
        # res[8] = prot.B_or(res[9], results[8])
        # res[4] = prot.B_or(res[5], results[4])
        # res[2] = prot.B_or(res[3], results[2])
        # res[1] = prot.B_or(res[3], x_1[30])
        # #fifth
        # res[0] = prot.B_or(res[1], results[0])
        # for i in range(0,64):
        #     results[i] = res[i]
        is_odd = prot.B_xor(results[prot.nbits - 2], results[prot.nbits - 3])

        for i in range(4, prot.nbits+1):
            is_odd = prot.B_xor(is_odd, results[prot.nbits - i])

        tmp1 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC, apply_scaling = False)
        tmp2 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC)
        is_odd = tfe.mul_AB(tmp2, is_odd)
        #is_odd = tfe.share_conversion_b_a(is_odd)

        #results = tfe.shares_conversion_b_a(results);
        exp = tfe.mul_AB(tmp1, results[0])
       # exp = results[0]
        b = (1 - exp)*(2**(prot.nbits-2)) + 1
        exp = exp*tmp2
        for i in range(1, prot.nbits-1):
            tmp00 = tfe.mul_AB(tmp1, results[i])
            #tmp00 = results[i]#tfe.share_conversion_b_a(results[i])
            exp += tmp00*tmp2
            b += (1-tmp00)*(2**(prot.nbits-2-i))
        b = tfe.truncate(b)
        b = tfe.truncate(b)
        b = tfe.truncate(x*b*4)
        b = tfe.truncate(b)

        b = tfe.polynomial(b, [2.223, -2.046, 0.8])
        # exp = exp - prot.fixedpoint_config.precision_fractional
        exp = prot.fixedpoint_config.precision_fractional - ((exp - prot.fixedpoint_config.precision_fractional)*0.5)

        exp_b = tfe.A2B(exp)

        exp_bs = []
        for i in range(prot.fixedpoint_config.precision_fractional, prot.fixedpoint_config.precision_fractional+5):
            exp_bs.append(tfe.bit_extract(exp_b, i))
        
        bs = []
        for i in range(0, len(exp_bs)):
            bs.append(tfe.mul_AB(tmp2, exp_bs[i]))

        
        ibs = []
        for i in range(0, len(exp_bs)):
            ibs.append(1 - bs[i])

        exp_sqrt = ((2**1) * bs[0] + ibs[0]);
        for i in range(1, len(bs)):
            exp_sqrt = exp_sqrt * ((2**(2**i)) * bs[i] + ibs[i])
        
        exp_sqrt = tfe.truncate(exp_sqrt)

        exp_sqrt_odd = exp_sqrt * (2**(0.5))

        exp_sqrt = exp_sqrt *  (  is_odd) + exp_sqrt_odd * (1 - is_odd)
        assert z.share_type == BOOLEAN
       
        res111 = b * exp_sqrt


        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            print("start")

            
            # reveal result

           # result = sess.run(res111.reveal())
            T1 = time.clock()
            times = []
            for i in range(1):
                 result = sess.run(res111.reveal())
                 T2 = time.clock()
                 times.append((T2-T1)*1000)
                 T1 = time.clock()

            print(np.average(times))
            print(np.min(result - ground_truth))
            np.save("test_tf_error.npy", result - ground_truth)
           # print(b.is_scaled)
            # np.testing.assert_allclose(
            #     result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            # )


if __name__ == "__main__":
  test_mat()