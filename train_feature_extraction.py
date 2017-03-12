import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time


#  Load traffic signs data.
# load the model from disk
n_classes = 43
epoch = 10
batch_Size = 128

filename = './train.p'
loaded_model = pickle.load(open(filename, 'rb'))
print("data is loaded!")

X,y = loaded_model['features'] , loaded_model['labels']
# Split data into training and validation sets.
X_train , X_valid , y_train , y_valid = train_test_split(X,y,test_size= 0.33,random_state=0)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64,None)
resized = tf.image.resize_images(x, (227, 227))
# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
#gettinh shape
shape = (fc7.get_shape().as_list()[-1],n_classes)
fc8_W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8_b = tf.Variable(tf.zeros(n_classes))
logits = tf.nn.xw_plus_b(fc7,fc8_W,fc8_b)


# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.


entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer()

training_operation = optimizer.minimize(loss, var_list=[fc8_W, fc8_b])

init_op = tf.global_variables_initializer()
correct_prediction = tf.argmax(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, labels), tf.float32))

def evaluate(X_data, y_data,sess):
    num_examples = X_data[0]
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, batch_Size):
        end = offset + batch_Size
        X_batch = X_data[offset:end]
        y_batch = y_data[offset:end]

        loss_val , acc_val = sess.run([loss,accuracy_operation] , feed_dict={x:X_batch,labels:y_batch})
        total_loss += (loss_val * X_batch.shape[0])
        total_accuracy += (acc_val * X_batch.shape[0])

    return total_loss/X_data.shape[0] , total_accuracy/X_data.shape[0]

#  Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epoch):
        #train it
        X_train , y_train = shuffle(X_train,y_train)
        t0 = time.time()
        for offset in range(0,X_train.shape[0],batch_Size):
            end = offset + batch_Size
            sess.run(training_operation, feed_dict={x:X_train[offset:end] , labels : y_train[offset:end]})
        get_loss, get_acc = evaluate(X_valid,y_valid)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("validation loss= {}".format(get_loss))
        print("validation accuracy= {}".format(get_acc))
        print("")

