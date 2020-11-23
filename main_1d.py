# Front matter: load libraries needed for training
import numpy as np
import tensorflow as tf
# from utils import *
import CVNNUtils
# import layersHDA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
plt.switch_backend('agg')

rad = 180 / np.pi
N = 65  # 2p+1; p=32
L = 1  # 2 angles

EPOCHS = 200

NUM_TRAIN = 18000 * 64  # 
NUM_VAL = 1800 * 64  # 
BATCH_SIZE = 6400*2  # 3200

load_len = 32  # 33


train_x = np.load("data/train_r.npy").reshape([NUM_TRAIN, load_len, 1])
train_y = np.load("data/train_doa.npy").reshape([NUM_TRAIN, L])

test_x = np.load("data/test_r.npy").reshape([NUM_VAL, load_len, 1])
test_y = np.load("data/test_doa.npy").reshape([NUM_VAL, L])


# 必须事先转化为特殊数据类型
# train_x = layersHDA.typeClone(train_x, 'Complex')
# MAE lr = 2e-4
alpha = 5e-4
lam = 0  #1e-8

# Set up graph for training
# x = tf.placeholder(tf.complex64, (None, 2080))
x = tf.placeholder(tf.complex64, (None, load_len, 1))
y = tf.placeholder(tf.float32, (None, L))
# keep_prob = tf.placeholder(tf.float32)
# Y1 = tf.placeholder(tf.float32, (None, 1), name="Y1")

# Select computational graph based on network input
fc0 = CVNNUtils.TASKs(x, out_nums=L, networkType='Complex')
Y1_layer = CVNNUtils.task_1d(fc0)


params = tf.trainable_variables()
reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])

Joint_Loss = rad * tf.losses.absolute_difference(y, Y1_layer) + lam * reg_loss
#Joint_Loss = tf.reduce_mean(tf.square(rad * (Y1_layer - y))) + lam * reg_loss


'''
y_target = tf.reshape(tf.concat([tf.cos(y), tf.sin(y)], 1), [-1, 1, 2])
Y1_layer = tf.reshape(Y1_layer, [-1, 2, 1])
Cos_dis = 1 - tf.matmul(y_target, Y1_layer)
Joint_Loss = tf.reduce_sum(Cos_dis) # + lam * reg_loss
'''
# T_y = tf.tan(y)
# loss_operation = tf.reduce_sum(tf.square(tf.tanh(logits) - tf.tanh(T_y))) + lam * reg_loss
'''
steps = int(NUM_TRAIN/BATCH_SIZE)
global_step = tf.Variable(0, trainable=False)
boundaries = [20, 100, 200]
learing_rates = [5e-5, 2e-5, 1e-5, 5e-6]

lr = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
'''
# optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha)
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

# Y1_op = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(Y1_Loss)
# Y2_op = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(Y2_Loss)

# training_operation = optimizer.minimize(Joint_Loss, global_step=global_step)
training_operation = optimizer.minimize(Joint_Loss)

saver = tf.train.Saver()

training_loss = np.zeros(int(np.ceil(EPOCHS * len(train_x) / BATCH_SIZE)))
# training_loss = np.zeros(int(np.ceil(EPOCHS * train_x.shape.as_list()[0] / BATCH_SIZE)))
test_loss = np.zeros(EPOCHS)

train_loss_count = 0

# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(train_x)
    # num_examples = train_x.shape[0]
    print("Training...")
    print()

    train_loss = []
    val_loss = []
    train_epochs = []

    for i in range(EPOCHS):
        X_train, y_train = shuffle(train_x, train_y)
        # X_train, y_train = train_x, train_y

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE

            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            '''
            batch_x_t = np.transpose(batch_x0, [0, 2, 1]).conj()
            batch_x = np.matmul(batch_x0, batch_x_t)
            batch_x = np.reshape(batch_x, ([BATCH_SIZE, 65, 65, 1]))
            '''

            sess.run(
                training_operation,
                feed_dict={
                    x: batch_x,
                    y: batch_y,
                    # global_step: i
                })

            training_loss[train_loss_count] = sess.run(
                Joint_Loss, # /BATCH_SIZE
                feed_dict={
                    x: batch_x,
                    y: batch_y,
                    # global_step: i
                    # keep_prob: 1.0
                })
            train_loss_count += 1

        test_loss[i] = sess.run(
            Joint_Loss,  # /NUM_VAL
            feed_dict={
                x: test_x,
                y: test_y,
                # keep_prob: 1.0
            })

        train_epochs.append(i)
        train_loss.append(training_loss[train_loss_count - 1])
        val_loss.append(test_loss[i])

        print("EPOCH {} ...".format(i + 1))
        # print('Learning rate: %f' % (sess.run(lr, feed_dict={global_step: i})))
        # print('Learning rate: %f' % (sess.run(optimizer._lr)))

        print("Training loss = {:.3f}".format(
            training_loss[train_loss_count - 1]))
        print("Test loss = {:.3f}".format(test_loss[i]))
        print()

    np.save("plot_x.npy", train_epochs)
    np.save("plot_yt.npy", train_loss)
    np.save("plot_yv.npy", val_loss)
    '''
    plt.plot(train_epochs, train_loss, label="train loss")
    plt.plot(train_epochs, val_loss, label="test loss")
    
    plt.annotate(
        "test loss = %.5s" % (val_loss[-1]), (train_epochs[-1], val_loss[-1]),
        (200, 0.3),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    

    plt.xlim((0, EPOCHS))
    plt.ylim((0, 1.2))  # plt.ylim((0, None))
    plt.title('Regression Model Training Loss')
    plt.ylabel('MAE Loss')
    plt.xlabel('Epoch')
    plt.legend(['training loss', 'test loss'], loc='upper right')
    plt.grid(True)
    plt.savefig("model_loss.eps")
    # plt.show()
    '''
    # saver.save(sess, "saveModel/MyModel")
    print("Model saveing...")
    # 保存参数
    builder = tf.saved_model.builder.SavedModelBuilder(
        "./saveModel_32/")  # PATH是保存路径
    # 保存整张网络及其变量，这种方法是可以保存多张网络的
    # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])

    # x 为输入tensor, keep_prob为dropout的prob tensor
    inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x)}

    # y 为最终需要的输出结果tensor
    outputs = {
        'output': tf.saved_model.utils.build_tensor_info(Y1_layer),
    }

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs, outputs, 'test_sig_name')

    builder.add_meta_graph_and_variables(sess, ['test_saved_model'],
                                         {'test_signature': signature})

    builder.save()  # 完成保存
    print("Model saved")
