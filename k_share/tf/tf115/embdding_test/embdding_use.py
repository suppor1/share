import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    aa = [1, 8, 5, 10, 7]
    bb = pd.Series(aa)
    print(bb.rank())

    with tf.variable_scope("v", reuse=tf.AUTO_REUSE):
        share_var = tf.get_variable("my_var", [20, 5])
    tensor = tf.convert_to_tensor([3, 19, 4])
    gt = tf.gather(share_var, tensor)
    lookEmbdding = tf.nn.embedding_lookup(share_var, tensor)

    # reduce
    mean = tf.reduce_mean(lookEmbdding, axis=0)
    mean1 = tf.reduce_mean(lookEmbdding, axis=-2)
    list = []
    for i in range(10):
        list.append(lookEmbdding)
    print(list)
    concat = tf.concat(list, axis=-1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(share_var)
        print("glorot_uniform_initializer:", sess.run(share_var))
        print("gather:", sess.run(gt))
        print("lookembdding:", sess.run(lookEmbdding))
        print("rank:", sess.run(tf.rank(lookEmbdding)))
        print("mean:", sess.run(mean))
        print("mean1:", sess.run(mean1))
        print("rand mean:", sess.run(tf.rank(mean)))
        print("concat: ", sess.run(concat))
