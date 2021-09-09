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
            tf.constant([[1, 2, 0.63], [4, 5, 9]]), share_type=ARITHMETIC
        )

        z = tfe.A2B(x)
        results = [];
        for i in range(0, prot.nbits):
            results.append(tfe.bit_extract(z, i))

        for i in range(2, prot.nbits):
            results[prot.nbits - i - 1] = prot.B_or(results[prot.nbits - i - 1], results[prot.nbits - i])

        is_odd = prot.B_xor(results[prot.nbits - 2], results[prot.nbits - 3])
        for i in range(4, prot.nbits+1):
             is_odd = prot.B_xor(is_odd, results[prot.nbits - i])

        tmp1 = tfe.define_constant(np.ones(x.shape), share_type=ARITHMETIC)

        exp = tfe.mul_AB(tmp1, results[0])
        for i in range(1, prot.nbits-1):
            exp += tfe.mul_AB(tmp1, results[i])
        


        assert z.share_type == BOOLEAN
       
        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            T1 = time.clock()
            # reveal result
            result = sess.run(exp.reveal())
            result1 = sess.run(is_odd.reveal())
            T2 = time.clock()
            print((T2-T1)*1000)
            print(result)
            # np.testing.assert_allclose(
            #     result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            # )
if __name__ == "__main__":
  test_a2b_private()