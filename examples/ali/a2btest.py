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
def test_a2b_private():
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 0.2], [4, 5, 9]]), share_type=ARITHMETIC
        )

        z = tfe.A2B(x)
        results = []
        for i in range(0, prot.nbits):
            results.append(tfe.bit_extract(z, i))

        for i in range(2, prot.nbits):
            results[prot.nbits - i - 1] = prot.B_or(results[prot.nbits - i - 1], results[prot.nbits - i])

        is_odd = prot.B_xor(results[prot.nbits - 2], results[prot.nbits - 3])
        for i in range(4, prot.nbits+1):
             is_odd = prot.B_xor(is_odd, results[prot.nbits - i])

        tmp1 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC, apply_scaling = False)
        tmp2 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC)
        is_odd = tfe.mul_AB(tmp2, is_odd)

        exp = tfe.mul_AB(tmp1, results[0])
        b = (1 - exp)*(2**(prot.nbits-2)) + 1
        exp = exp*tmp2
        for i in range(1, prot.nbits-1):
            tmp00 = tfe.mul_AB(tmp1, results[i])
            exp += tmp00*tmp2
            b += (1-tmp00)*(2**(prot.nbits-2-i))
        b = tfe.truncate(b)
        b = tfe.truncate(b)
        b = tfe.truncate(x*b*4)
        b = tfe.truncate(b)

        b = tfe.polynomial(b, [2.223, -2.046, 0.8277])
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

        exp_sqrt = exp_sqrt *  ( is_odd) + exp_sqrt_odd * (1 - is_odd)
        assert z.share_type == BOOLEAN
       
        res = b * exp_sqrt
        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            T1 = time.clock()
            # reveal result
            result = sess.run(res.reveal())
            T2 = time.clock()
            print((T2-T1)*1000)
            print(result)
            print(b.is_scaled)
            # np.testing.assert_allclose(
            #     result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            # )
if __name__ == "__main__":
  test_a2b_private()