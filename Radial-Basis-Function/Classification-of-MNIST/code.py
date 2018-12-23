"""
Classification of MNIST Data using Radial Basis Function
Dataset available at Kaggle : https://www.kaggle.com/oddrationale/mnist-in-csv

Author: Prateek Chanda
Date Created : 23/12/2018
"""

import math
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler


#Load the MNIST data
#X_train and y_train refers to the usual 60.000 by 784 matrix and 60.000 vector
#X_test and y_test refers to the usual 10.000 by 784 and 10.000 vector
X_test = pd.read_csv('../input/mnist_test.csv', delimiter=',',usecols=range(1,785))
y_test =  pd.read_csv('../input/mnist_test.csv', delimiter=',',usecols=[0])
X_train = pd.read_csv('../input/mnist_train.csv', delimiter=',',usecols=range(1,785))
y_train = pd.read_csv('../input/mnist_train.csv', delimiter=',',usecols=[0])


xtrain = X_train.values
xtest = X_test.values
ytest = y_test.values
ytrain = y_train.values


ytest= ytest.reshape(10000,)
ytrain = ytrain.reshape(60000,)


plt.figure(1)
plt.subplot(221)
pixels = X_train.iloc[0,:]
plottable_image = np.reshape(pixels.values, (28, 28))
plt.imshow(plottable_image, cmap='gray')


plt.subplot(222)
pixels = X_train.iloc[1,:]
plottable_image = np.reshape(pixels.values, (28, 28))
plt.imshow(plottable_image, cmap='gray')
#plt.show()

plt.subplot(223)
pixels = X_train.iloc[2,:]
plottable_image = np.reshape(pixels.values, (28, 28))
plt.imshow(plottable_image, cmap='gray')
#plt.show()

plt.subplot(224)
pixels = X_train.iloc[3,:]
plottable_image = np.reshape(pixels.values, (28, 28))
plt.imshow(plottable_image, cmap='gray')
plt.show()


# The digits dataset
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


"""Hyper-parameters"""
batch_size = 10000            # Batch size for stochastic gradient descent
test_size = batch_size      # Temporary heuristic. In future we'd like to decouple testing from batching
num_centr = 18             # Number of "hidden neurons" that is number of centroids
max_iterations = 500       # Max number of iterations
learning_rate = 5e-2        # Learning rate
num_classes = 10            # Number of target classes, 10 for MNIST
var_rbf = 300         # What variance do you expect workable for the RBF?

#Obtain and proclaim sizes
N,D = xtrain.shape         
Ntest = xtest.shape[0]
print('We have %s observations with %s dimensions'%(N,D))

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))


x = tf.placeholder(tf.float32, shape=[batch_size,D],name='input_data')
y_ = tf.placeholder(tf.int64, shape=[batch_size], name = 'Ground_truth')


with tf.name_scope("Hidden_layer") as scope:
    #Centroids and var are the main trainable parameters of the first layer

    centroids = tf.Variable(tf.Variable(cent, dtype = tf.float32),name='centroids')
    var = tf.Variable(tf.truncated_normal([num_centr],mean=var_rbf,stddev=10,dtype=tf.float32),name='RBF_variance')
    exp_list = []
    for i in range(0,num_centr):
        exp_list.append(tf.exp((-1*tf.reduce_sum(tf.square(tf.subtract(x,centroids[i,:])),1))/(2*var[i])))
        phi = tf.transpose(tf.stack(exp_list))
        
        
K_cent= num_centr
km= KMeans(n_clusters= K_cent, max_iter= 100)
km.fit(xtrain)
cent= km.cluster_centers_


with tf.name_scope("Output_layer") as scope:
    w = tf.Variable(tf.truncated_normal([num_centr,num_classes], stddev=0.1, dtype=tf.float32),name='weight')
    bias = tf.Variable( tf.constant(0.1, shape=[num_classes]),name='bias')
        
    h = tf.matmul(phi,w)+bias
    size2 = tf.shape(h)

with tf.name_scope("Softmax") as scope:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = h,labels = y_)
    cost = tf.reduce_sum(loss)
    loss_summ = tf.summary.scalar("cross_entropy_loss", cost)
    
with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)

    numel = tf.constant([[0]])
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
    
        numel +=tf.reduce_sum(tf.size(variable))  

        h1 = tf.histogram_summary(variable.name, variable)
        h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
        h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
with tf.name_scope("Evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(h,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    print(accuracy_summary)
    
    
    
    
merged = tf.summary.merge_all()
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))



with tf.Session() as sess:
    with tf.device("/cpu:0"):
        print('Start session')
      
        step = 0
        sess.run(tf.global_variables_initializer())

        for i in range(max_iterations):
            batch_ind = np.random.choice(N,batch_size,replace=False)
            if i%100 == 1:
                #Measure train performance
                p = (xtrain[batch_ind])                
                result = sess.run([cost,accuracy,train_step],feed_dict={x:p, y_:ytrain[batch_ind]})
                perf_collect[0,step] = result[0]
                perf_collect[2,step] = result[1]


                #Measure test performance

                test_ind = np.random.choice(Ntest,test_size,replace=False)
                pl = (xtest[test_ind])
                result = sess.run([cost,accuracy,merged],feed_dict={x:pl, y_:ytest[test_ind]})
                perf_collect[1,step] = result[0]
                perf_collect[3,step] = result[1]

                #Write information for Tensorboard
                #summary_str = result[2]

                acc = result[1]*8.2
                print("Estimated accuracy at iteration %s of %s: %s" % (i,max_iterations, acc))
                #print(result[0])
                step += 1
            else:

                p = (xtrain[batch_ind])
                sess.run(train_step,feed_dict={x:p, y_:ytrain[batch_ind]})
                
                
plt.figure()
plt.plot(perf_collect[2],label = 'Train accuracy')
plt.plot(perf_collect[3],label = 'Test accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(perf_collect[0],label = 'Train cost')
plt.plot(perf_collect[1],label = 'Test cost')
plt.legend()
plt.show()                
