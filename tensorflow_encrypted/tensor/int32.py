from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any, Tuple, Type
from .tensor import AbstractTensor, AbstractVariable, AbstractConstant, AbstractPlaceholder
from .factory import AbstractFactory
from .native import NativeTensor

from ..config import run


class Int32Tensor(AbstractTensor):

    modulus = 2**31
    int_type = tf.int32

    def __init__(self, value: Union[np.ndarray, tf.Tensor]) -> None:
        self.value = value

    @classmethod
    def from_native(cls, value: Union[np.ndarray, tf.Tensor]) -> 'Int32Tensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return cls(value)

    @classmethod
    def from_same(cls, value: 'Int32Tensor') -> 'Int32Tensor':
        assert isinstance(value, Int32Tensor), type(value)
        return cls(value.value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={}, tag: Optional[str]=None) -> 'Int32Tensor':
        concrete_value = run(sess, self.value, feed_dict=feed_dict, tag=tag)
        return Int32Tensor.from_native(concrete_value)

    def to_int32(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    @staticmethod
    def sample_uniform(shape: List[int]) -> 'Int32Tensor':
        # TODO[Morten] what should maxval be (account for negative numbers)?
        return Int32Tensor(tf.random_uniform(shape=shape, dtype=tf.int32, maxval=2**31 - 1))

    def __repr__(self) -> str:
        return 'int32.Tensor(shape={})'.format(self.shape)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        if self.value is None:
            raise Exception("Can't call 'shape' on a empty tensor")

        return self.value.shape

    def __add__(self, other: 'Int32Tensor') -> 'Int32Tensor':
        return self.add(other)

    def __sub__(self, other: 'Int32Tensor') -> 'Int32Tensor':
        return self.sub(other)

    def __mul__(self, other: 'Int32Tensor') -> 'Int32Tensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'Int32Tensor':
        return self.mod(k)

    def add(self, other: 'Int32Tensor') -> 'Int32Tensor':
        x, y = _lift(self), _lift(other)
        return Int32Tensor(x.value + y.value)

    def sub(self, other: 'Int32Tensor') -> 'Int32Tensor':
        x, y = _lift(self), _lift(other)
        return Int32Tensor(x.value - y.value)

    def mul(self, other: 'Int32Tensor') -> 'Int32Tensor':
        x, y = _lift(self), _lift(other)
        return Int32Tensor(x.value * y.value)

    def dot(self, other: 'Int32Tensor') -> 'Int32Tensor':
        x, y = _lift(self), _lift(other)
        return Int32Tensor(tf.matmul(x.value, y.value))

    def im2col(self, h_filter, w_filter, padding, strides) -> 'Int32Tensor':
        raise NotImplementedError()

    def conv2d(self, other, strides, padding='SAME') -> 'Int32Tensor':
        raise NotImplementedError()

    def mod(self, k: int) -> 'Int32Tensor':
        return Int32Tensor(self.value % k)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'Int32Tensor':
        return Int32Tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'Int32Tensor':
        return Int32Tensor(tf.strided_slice(self.value, *args, **kwargs))

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'Int32Tensor':
        return Int32Tensor(tf.reshape(self.value, axes))

    @staticmethod
    def stack(xs: List['Int32Tensor'], axis: int = 0) -> 'Int32Tensor':
        assert all(isinstance(x, Int32Tensor) for x in xs)
        return Int32Tensor(tf.stack([x.value for x in xs], axis=axis))

    @staticmethod
    def concat(xs: List['Int32Tensor'], axis: int) -> 'Int32Tensor':
        assert all(isinstance(x, Int32Tensor) for x in xs)
        return Int32Tensor(tf.concat([x.value for x in xs], axis=axis))

    def binarize(self, prime=37) -> NativeTensor:
        BITS = 32
        assert prime > BITS, prime

        final_shape = [1] * len(self.shape) + [BITS]
        bitwidths = tf.range(BITS, dtype=tf.int32)
        bitwidths = tf.reshape(bitwidths, final_shape)

        val = tf.expand_dims(self.value, -1)
        val = tf.bitwise.bitwise_and(tf.bitwise.right_shift(val, bitwidths), 1)

        return NativeTensor.from_native(val, prime)


def _lift(x: Union[Int32Tensor, int]) -> Int32Tensor:
    # TODO[Morten] support other types of `x`

    if isinstance(x, Int32Tensor):
        return x

    if type(x) is int:
        return Int32Tensor.from_native(np.array([x]))

    raise TypeError("Unsupported type {}".format(type(x)))


class Int32Constant(Int32Tensor, AbstractConstant):

    def __init__(self, value: Union[tf.Tensor, np.ndarray]) -> None:
        super(Int32Constant, self).__init__(tf.constant(value, dtype=tf.int32))

    def __repr__(self) -> str:
        return 'int32.Constant(shape={})'.format(self.shape)


class Int32Placeholder(Int32Tensor, AbstractPlaceholder):

    def __init__(self, shape: List[int]) -> None:
        placeholder = tf.placeholder(tf.int32, shape=shape)
        super(Int32Placeholder, self).__init__(placeholder)
        self.placeholder = placeholder

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def __repr__(self) -> str:
        return 'int32.Placeholder(shape={})'.format(self.shape)


class Int32Variable(Int32Tensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
        variable = tf.Variable(initial_value, dtype=tf.int32, trainable=False)
        value = variable.read_value()

        super(Int32Variable, self).__init__(value)
        self.variable = variable
        self.initializer = variable.initializer

    def __repr__(self) -> str:
        return 'int32.Variable(shape={})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_same(self, value: Int32Tensor) -> tf.Operation:
        assert isinstance(value, Int32Tensor), type(value)
        return tf.assign(self.variable, value.value).op


class Int32Factory(AbstractFactory):
    @property
    def Tensor(self) -> Type[Int32Tensor]:
        return Int32Tensor

    @property
    def Constant(self) -> Type[Int32Constant]:
        return Int32Constant

    @property
    def Variable(self) -> Type[Int32Variable]:
        return Int32Variable

    def Placeholder(self, shape: List[int]) -> Int32Placeholder:
        return Int32Placeholder(shape)

    @property
    def modulus(self) -> int:
        return Int32Tensor.modulus
