"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
from common import DataOwner
#from common import LogisticRegression
import tensorflow as tf
from tf_encrypted.protocol.aby3.model.logistic_regression import LogisticRegression
from common import ModelOwner
from tf_encrypted.protocol.aby3 import ABY3
num_features = 2
training_set_size = 2000
test_set_size = 100
batch_size = 100
num_batches = (training_set_size // batch_size) * 10

model_owner = ModelOwner("model-owner")
data_owner_0 = DataOwner(
    "data-owner-0", num_features, training_set_size, test_set_size, batch_size // 2
)
data_owner_1 = DataOwner(
    "data-owner-1", num_features, training_set_size, test_set_size, batch_size // 2
)
prot = ABY3()
tfe.set_protocol(prot)

# x_train_0, y_train_0 = data_owner_0.provide_training_data()
# x_train_1, y_train_1 = data_owner_1.provide_training_data()

# x_test_0, y_test_0 = data_owner_0.provide_testing_data()
# x_test_1, y_test_1 = data_owner_1.provide_testing_data()

x = tfe.define_private_variable(tf.ones(shape=(2, 2)) )
y = tfe.define_private_variable(tf.ones(shape=(2, 1)))

x_train = x#tfe.concat([x_train_0, x_train_1], axis=0)
y_train = y#tfe.concat([y_train_0, y_train_1], axis=0)

model = LogisticRegression(num_features)
reveal_weights_op = model_owner.receive_weights(model.weights)

with tfe.Session() as sess:
    sess.run(
        [
            tfe.global_variables_initializer()
        ],
        tag="init",
    )

    model.fit(sess, x_train, y_train, num_batches)
    # TODO(Morten)
    # each evaluation results in nodes for a forward pass being added to the graph;
    # maybe there's some way to avoid this, even if it means only if the shapes match
 #   model.evaluate(sess, x_test_0, y_test_0, data_owner_0)
    #   model.evaluate(sess, x_test_1, y_test_1, data_owner_1)

    sess.run(reveal_weights_op, tag="reveal")
