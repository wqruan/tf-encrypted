def logistic_regression():
    prot = ABY3()
    tfe.set_protocol(prot)
# define inputs
    x = tfe.define_private_variable(x_raw, name="x")
    y = tfe.define_private_variable(y_raw, name="y")

# define initial weights
    w = tfe.define_private_variable(tf.random_uniform([10, 1], -0.01, 0.01),name="w")
    learning_rate = 0.01
    with tf.name_scope("forward"):
        out = tfe.matmul(x, w) + b
        y_hat = tfe.sigmoid(out)
    with tf.name_scope("loss-grad"):
        dy = y_hat - y
    batch_size = x.shape.as_list()[0]
    with tf.name_scope("backward"):
        dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
        assign_ops = [tfe.assign(w, w - dw * learning_rate)]
    with tfe.Session() as sess:
# initialize variables
        sess.run(tfe.global_variables_initializer())
        for i in range(1):
            print(2134)
            sess.run(assign_ops)