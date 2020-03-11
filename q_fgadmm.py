# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:11:10 2020

@author: saahmed
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import gc

import matplotlib.pyplot as plt

num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train = np.reshape(x_train, (x_train.shape[0], -1))/255
x_test = np.reshape(x_test, (x_test.shape[0], -1))/255

n1 = 128
n2 = 64
R = tf.Variable(tf.zeros([1]))

#User 1

x1 = tf.placeholder(tf.float32,shape=[None,784])
y_true1 = tf.placeholder(tf.float32, [None, 10])

W11 = tf.Variable(tf.truncated_normal([784, n1], stddev=0.1))
lambda_W11 = tf.zeros([784, n1])
b11 = tf.Variable(tf.zeros([n1]))
lambda_b11 = tf.zeros([n1])

W12 = tf.Variable(tf.truncated_normal([n1, n2], stddev=0.1))
lambda_W12 = tf.zeros([n1,n2])
b12 = tf.Variable(tf.zeros([n2]))
lambda_b12 = tf.zeros([n2])

W13 = tf.Variable(tf.truncated_normal([n2, 10], stddev=0.1))
lambda_W13 = tf.zeros([n2, 10])
b13 = tf.Variable(tf.zeros([10]))
lambda_b13 = tf.zeros([10])

y11 = tf.nn.relu(tf.matmul(x1, W11) + b11)
y12 = tf.nn.relu(tf.matmul(y11, W12) + b12)

ylogits1 = tf.matmul(y12, W13) + b13
y1 = tf.nn.softmax(ylogits1)

#User 2

x2 = tf.placeholder(tf.float32,shape=[None,784])
y_true2 = tf.placeholder(tf.float32, [None, 10])

W21 = tf.Variable(tf.truncated_normal([784, n1], stddev=0.1))
lambda_W21 = tf.zeros([784, n1])
b21 = tf.Variable(tf.zeros([n1]))
lambda_b21 = tf.zeros([n1])

W22 = tf.Variable(tf.truncated_normal([n1, n2], stddev=0.1))
lambda_W22 = tf.zeros([n1,n2])
b22 = tf.Variable(tf.zeros([n2]))
lambda_b22 = tf.zeros([n2])

W23 = tf.Variable(tf.truncated_normal([n2, 10], stddev=0.1))
lambda_W23 = tf.zeros([n2, 10])
b23 = tf.Variable(tf.zeros([10]))
lambda_b23 = tf.zeros([10])

y21 = tf.nn.relu(tf.matmul(x2, W21) + b21)
y22 = tf.nn.relu(tf.matmul(y21, W22) + b22)

ylogits2 = tf.matmul(y22, W23) + b23
y2 = tf.nn.softmax(ylogits2)

#User 3

x3 = tf.placeholder(tf.float32,shape=[None,784])
y_true3 = tf.placeholder(tf.float32, [None, 10])

W31 = tf.Variable(tf.truncated_normal([784, n1], stddev=0.1))
lambda_W31 = tf.zeros([784, n1])
b31 = tf.Variable(tf.zeros([n1]))
lambda_b31 = tf.zeros([n1])

W32 = tf.Variable(tf.truncated_normal([n1, n2], stddev=0.1))
lambda_W32 = tf.zeros([n1,n2])
b32 = tf.Variable(tf.zeros([n2]))
lambda_b32 = tf.zeros([n2])

W33 = tf.Variable(tf.truncated_normal([n2, 10], stddev=0.1))
lambda_W33 = tf.zeros([n2, 10])
b33 = tf.Variable(tf.zeros([10]))
lambda_b33 = tf.zeros([10])

y31 = tf.nn.relu(tf.matmul(x3, W31) + b31)
y32 = tf.nn.relu(tf.matmul(y31, W32) + b32)

ylogits3 = tf.matmul(y32, W33) + b33
y3 = tf.nn.softmax(ylogits3)

#User 4

x4 = tf.placeholder(tf.float32,shape=[None,784])
y_true4 = tf.placeholder(tf.float32, [None, 10])

W41 = tf.Variable(tf.truncated_normal([784, n1], stddev=0.1))
lambda_W41 = tf.zeros([784, n1])
b41 = tf.Variable(tf.zeros([n1]))
lambda_b41 = tf.zeros([n1])

W42 = tf.Variable(tf.truncated_normal([n1, n2], stddev=0.1))
lambda_W42 = tf.zeros([n1,n2])
b42 = tf.Variable(tf.zeros([n2]))
lambda_b42 = tf.zeros([n2])

W43 = tf.Variable(tf.truncated_normal([n2, 10], stddev=0.1))
lambda_W43 = tf.zeros([n2, 10])
b43 = tf.Variable(tf.zeros([10]))
lambda_b43 = tf.zeros([10])

y41 = tf.nn.relu(tf.matmul(x4, W41) + b41)
y42 = tf.nn.relu(tf.matmul(y41, W42) + b42)

ylogits4 = tf.matmul(y42, W43) + b43
y4 = tf.nn.softmax(ylogits4)

cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true1, logits= ylogits1))*100
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true2, logits= ylogits2))*100
cross_entropy3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true3, logits= ylogits3))*100
cross_entropy4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true4, logits= ylogits4))*100
cross_entropy = cross_entropy1 + cross_entropy3 + cross_entropy2 + cross_entropy4

rho = tf.constant(1e0)
alpha=1.0
regW = rho/2*tf.nn.l2_loss(W11-W21)+ rho/2*tf.nn.l2_loss(W12-W22)\
+ rho/2*tf.nn.l2_loss(W13-W23)\
+ rho/2*tf.nn.l2_loss(W21-W31)\
+ rho/2*tf.nn.l2_loss(W22-W32)\
+ rho/2*tf.nn.l2_loss(W23-W33)\
+ rho/2*tf.nn.l2_loss(W31-W41)\
+ rho/2*tf.nn.l2_loss(W32-W42)\
+ rho/2*tf.nn.l2_loss(W33-W43)
regB = rho/2*tf.nn.l2_loss(b11-b21)+ rho/2*tf.nn.l2_loss(b12-b22)\
+ rho/2*tf.nn.l2_loss(b13-b23)\
+ rho/2*tf.nn.l2_loss(b21-b31)\
+ rho/2*tf.nn.l2_loss(b22-b32)\
+ rho/2*tf.nn.l2_loss(b23-b33)\
+ rho/2*tf.nn.l2_loss(b31-b41)\
+ rho/2*tf.nn.l2_loss(b32-b42)\
+ rho/2*tf.nn.l2_loss(b33-b43)
reg = regW+regB

lambdas_w =tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W11,(W11-W21))))+\
tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W12,(W12-W22))))+\
tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W13,(W13-W23))))\
+tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W21,(W21-W31))))\
+tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W22,(W22-W32))))\
+tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W23,(W23-W33))))\
+tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W31,(W31-W41))))\
+tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W32,(W32-W42))))\
+tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W33,(W33-W43))))
lambdas_b = tf.tensordot(tf.transpose(lambda_b11),(b11-b21),1)\
+tf.tensordot(tf.transpose(lambda_b12),(b12-b22),1)\
+tf.tensordot(tf.transpose(lambda_b13),(b13-b23),1)\
+tf.tensordot(tf.transpose(lambda_b21),(b21-b31),1)\
+tf.tensordot(tf.transpose(lambda_b22),(b22-b32),1)\
+tf.tensordot(tf.transpose(lambda_b23),(b23-b33),1)\
+tf.tensordot(tf.transpose(lambda_b31),(b31-b41),1)\
+tf.tensordot(tf.transpose(lambda_b32),(b32-b42),1)\
+tf.tensordot(tf.transpose(lambda_b33),(b33-b43),1)\

lambdas = lambdas_w + lambdas_b
loss = cross_entropy + reg + lambdas
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_W1 = optimizer.minimize(loss, var_list=[W11,W12, W13, b11, b12, b13])
train_W3 = optimizer.minimize(loss, var_list=[W31,W32,W33,b31,b32,b33])
train_W2 = optimizer.minimize(loss, var_list=[W21,W22,W23,b21,b22,b23])
train_W4 = optimizer.minimize(loss, var_list=[W41,W42,W43, b41,b42,b43])

def find_max(WW1):
    
    R1_max = tf.reduce_max(WW1)
    myMax=R1_max.eval()
    
    R1_min = tf.reduce_min(WW1)
    myMin=R1_min.eval()
    
    return myMax, myMin

init = tf.global_variables_initializer()

def mini_batches(X, Y, mb_size = 100):

    m = X.shape[0]

    perm = list(np.random.permutation(m))
    #perm = perm_init[0:100]
    X_temp = X[perm,:]
    Y_temp = Y[perm,:].reshape((m, Y.shape[1]))
    
    X_r = X_temp[0:mb_size,:]
    Y_r = Y_temp[0:mb_size,:]
    return X_r,Y_r

x_train11 = x_train[:1000]
y_train11 = y_train[:1000]

x_train12 = x_train[1000:2000]
y_train12 = y_train[1000:2000]

x_train13 = x_train[2000:3000]
y_train13 = y_train[2000:3000]

x_train14 = x_train[3000:4000]
y_train14 = y_train[3000:4000]

x_test1 = x_test
y_test1 = y_test

import time

start = time.time()

abs_weights_diff = []
abs_biases_diff = []
Train_Acc1 = []
Test_Acc1 = []
Train_Acc2 = []
Test_Acc2 = []
Train_Acc3 = []
Test_Acc3 = []
Train_Acc4 = []
Test_Acc4 = []
CrE_Train = []
CrE_Test = []
b=8
tau=1/(2**b-1)

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(5):
        #print(i)
        batch_x1 , batch_y1 = mini_batches(x_train11,y_train11, 100)
        batch_x2 , batch_y2 = mini_batches(x_train12,y_train12, 100)
        batch_x3 , batch_y3 = mini_batches(x_train13,y_train13, 100)
        batch_x4 , batch_y4 = mini_batches(x_train14,y_train14, 100)
        
        sess.run(train_W1,feed_dict={x1:batch_x1,y_true1:batch_y1})
        sess.run(train_W3,feed_dict={x3:batch_x3,y_true3:batch_y3})
        
        #User 1
        [max_value, min_value]=find_max(W11)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q11=(W11+R[0])/(2*tau*R[0])+1/2
        Q11=tf.math.floor(Q11)#.eval())
        Q11= 2*tau*Q11*R[0]-R[0]
        sess.run(tf.assign(W11,Q11))
        #print(W11[0,0:5].eval())
        
        [max_value, min_value]=find_max(W12)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q12=(W12+R[0])/(2*tau*R[0])+1/2
        Q12=tf.math.floor(Q12)#.eval())
        Q12= 2*tau*Q12*R[0]-R[0]
        sess.run(tf.assign(W12,Q12))
        
        [max_value, min_value]=find_max(W13)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q13=(W13+R[0])/(2*tau*R[0])+1/2
        Q13=tf.math.floor(Q13)#.eval())
        Q13= 2*tau*Q13*R[0]-R[0]
        sess.run(tf.assign(W13,Q13))
        
         #User 3
        [max_value, min_value]=find_max(W31)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q31=(W31+R[0])/(2*tau*R[0])+1/2
        Q31=tf.math.floor(Q31)#.eval())
        Q31= 2*tau*Q31*R[0]-R[0]
        sess.run(tf.assign(W31,Q31))
        
        [max_value, min_value]=find_max(W32)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q32=(W32+R[0])/(2*tau*R[0])+1/2
        Q32=tf.math.floor(Q32)#.eval())
        Q32= 2*tau*Q32*R[0]-R[0]
        sess.run(tf.assign(W32,Q32))
        
        [max_value, min_value]=find_max(W33)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q33=(W33+R[0])/(2*tau*R[0])+1/2
        Q33=tf.math.floor(Q33)#.eval())
        Q33= 2*tau*Q33*R[0]-R[0]
        sess.run(tf.assign(W33,Q33))
      
        sess.run(train_W2,feed_dict={x2:batch_x2,y_true2:batch_y2})
        sess.run(train_W4,feed_dict={x4:batch_x4,y_true4:batch_y4})
        
        #User 2
        [max_value, min_value]=find_max(W21)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q21=(W21+R[0])/(2*tau*R[0])+1/2
        Q21=tf.math.floor(Q21)#.eval())
        Q21= 2*tau*Q21*R[0]-R[0]
        sess.run(tf.assign(W21,Q21))
        
        [max_value, min_value]=find_max(W22)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q22=(W22+R[0])/(2*tau*R[0])+1/2
        Q22=tf.math.floor(Q22)#.eval())
        Q22= 2*tau*Q22*R[0]-R[0]
        sess.run(tf.assign(W22,Q22))
        
        [max_value, min_value]=find_max(W23)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q23=(W23+R[0])/(2*tau*R[0])+1/2
        Q23=tf.math.floor(Q23)#.eval())
        Q23= 2*tau*Q23*R[0]-R[0]
        sess.run(tf.assign(W23,Q23))
        
        #User 4
        [max_value, min_value]=find_max(W41)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q41=(W41+R[0])/(2*tau*R[0])+1/2
        Q41=tf.math.floor(Q41)#.eval())
        Q41= 2*tau*Q41*R[0]-R[0]
        sess.run(tf.assign(W41,Q41))
        
        [max_value, min_value]=find_max(W42)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q42=(W42+R[0])/(2*tau*R[0])+1/2
        Q42=tf.math.floor(Q42)#.eval())
        Q42= 2*tau*Q42*R[0]-R[0]
        sess.run(tf.assign(W42,Q42))
        
        [max_value, min_value]=find_max(W43)
        sess.run(tf.assign(R[0],max(abs(min_value), abs(max_value))))
        Q43=(W43+R[0])/(2*tau*R[0])+1/2
        Q43=tf.math.floor(Q43)#.eval())
        Q43= 2*tau*Q43*R[0]-R[0]
        sess.run(tf.assign(W43,Q43))
        
        lambda_W11 = lambda_W11 + rho*(W11 - W21)
        lambda_b11 = lambda_b11 + rho*(b11 - b21)
        lambda_W12 = lambda_W12 + rho*(W12 - W22)
        lambda_b12 = lambda_b12 + rho*(b12 - b22)
        lambda_W13 = lambda_W13 + rho*(W13 - W23)
        lambda_b13 = lambda_b13 + rho*(b13 - b23)
            
        lambda_W21 = lambda_W21 + rho*(W21 - W31)
        lambda_b21 = lambda_b21 + rho*(b21 - b31)
        lambda_W22 = lambda_W22 + rho*(W22 - W32)
        lambda_b22 = lambda_b22 + rho*(b22 - b32)
        lambda_W23 = lambda_W23 + rho*(W23 - W33)
        lambda_b23 = lambda_b23 + rho*(b23 - b33)
            
        lambda_W31 = lambda_W31 + rho*(W31 - W41)
        lambda_b31 = lambda_b31 + rho*(b31 - b41)
        lambda_W32 = lambda_W32 + rho*(W32 - W42)
        lambda_b32 = lambda_b32 + rho*(b32 - b42)
        lambda_W33 = lambda_W33 + rho*(W33 - W43)
        lambda_b33 = lambda_b33 + rho*(b33 - b43)
        
    
        matches1 = tf.equal(tf.argmax(y1,1),tf.argmax(y_true1,1))
        acc1 = tf.reduce_mean(tf.cast(matches1,tf.float32))
    
        matches2 = tf.equal(tf.argmax(y2,1),tf.argmax(y_true2,1))
        acc2 = tf.reduce_mean(tf.cast(matches2,tf.float32))
    
        matches3 = tf.equal(tf.argmax(y3,1),tf.argmax(y_true3,1))
        acc3 = tf.reduce_mean(tf.cast(matches3,tf.float32))
    
        matches4 = tf.equal(tf.argmax(y4,1),tf.argmax(y_true4,1))
        acc4 = tf.reduce_mean(tf.cast(matches4,tf.float32))
    
        TrainAcc1, TrainLoss1 = sess.run([acc1, cross_entropy1] ,feed_dict={x1:x_train11, y_true1:y_train11})
        Train_Acc1.append(TrainAcc1)
        TestAcc1, TestLoss1 = sess.run([acc1,cross_entropy1] ,feed_dict={x1:x_test1,y_true1:y_test1})
        Test_Acc1.append(TestAcc1)
        TrainAcc2, TrainLoss2 = sess.run([acc2,cross_entropy2],feed_dict={x2:x_train12, y_true2:y_train12})
        Train_Acc2.append(TrainAcc2)
        TestAcc2, TestLoss2 = sess.run([acc2,cross_entropy2],feed_dict={x2:x_test1, y_true2:y_test1})
        Test_Acc2.append(TestAcc2)
        TrainAcc3, TrainLoss3 = sess.run([acc3,cross_entropy3],feed_dict={x3:x_train13, y_true3:y_train13})
        Train_Acc3.append(TrainAcc3)
        TestAcc3, TestLoss3 = sess.run([acc3,cross_entropy3],feed_dict={x3:x_test1, y_true3:y_test1})
        Test_Acc3.append(TestAcc3)
        TrainAcc4, TrainLoss4 = sess.run([acc4,cross_entropy4],feed_dict={x4:x_train14, y_true4:y_train14})
        Train_Acc4.append(TrainAcc4)
        TestAcc4, TestLoss4 = sess.run([acc4,cross_entropy4],feed_dict={x4:x_test1, y_true4:y_test1})
        Test_Acc4.append(TestAcc4)
        TrainLoss =  TrainLoss1 + TrainLoss2 + TrainLoss3 + TrainLoss4
        CrE_Train.append(TrainLoss)
        TestLoss = TestLoss1 + TestLoss2 + TestLoss3 + TestLoss4
        CrE_Test.append(TestLoss)
        avgAcc_Train = [(a+b+c+d)/4 for a,b,c,d in zip(Train_Acc1, Train_Acc2,Train_Acc3, Train_Acc4)]
        avgAcc_Test = [(a+b+c+d)/4 for a,b,c,d in zip(Test_Acc1, Test_Acc2, Test_Acc3, Test_Acc4)]
        print(avgAcc_Test)
        
        regW = rho/2*tf.nn.l2_loss(W11-W21)+ rho/2*tf.nn.l2_loss(W12-W22)\
        + rho/2*tf.nn.l2_loss(W13-W23)\
        + rho/2*tf.nn.l2_loss(W21-W31)\
        + rho/2*tf.nn.l2_loss(W22-W32)\
        + rho/2*tf.nn.l2_loss(W23-W33)\
        + rho/2*tf.nn.l2_loss(W31-W41)\
        + rho/2*tf.nn.l2_loss(W32-W42)\
        + rho/2*tf.nn.l2_loss(W33-W43)
        regB = rho/2*tf.nn.l2_loss(b11-b21)+ rho/2*tf.nn.l2_loss(b12-b22)\
        + rho/2*tf.nn.l2_loss(b13-b23)\
        + rho/2*tf.nn.l2_loss(b21-b31)\
        + rho/2*tf.nn.l2_loss(b22-b32)\
        + rho/2*tf.nn.l2_loss(b23-b33)\
        + rho/2*tf.nn.l2_loss(b31-b41)\
        + rho/2*tf.nn.l2_loss(b32-b42)\
        + rho/2*tf.nn.l2_loss(b33-b43)
        reg = regW+regB

        lambdas_w =tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W11,(W11-W21))))+\
        tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W12,(W12-W22))))+\
        tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W13,(W13-W23))))\
        +tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W21,(W21-W31))))\
        +tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W22,(W22-W32))))\
        +tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W23,(W23-W33))))\
        +tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W31,(W31-W41))))\
        +tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W32,(W32-W42))))\
        +tf.reduce_sum(tf.compat.v1.linalg.diag_part(tf.math.multiply(lambda_W33,(W33-W43))))
        lambdas_b = tf.tensordot(tf.transpose(lambda_b11),(b11-b21),1)\
        +tf.tensordot(tf.transpose(lambda_b12),(b12-b22),1)\
        +tf.tensordot(tf.transpose(lambda_b13),(b13-b23),1)\
        +tf.tensordot(tf.transpose(lambda_b21),(b21-b31),1)\
        +tf.tensordot(tf.transpose(lambda_b22),(b22-b32),1)\
        +tf.tensordot(tf.transpose(lambda_b23),(b23-b33),1)\
        +tf.tensordot(tf.transpose(lambda_b31),(b31-b41),1)\
        +tf.tensordot(tf.transpose(lambda_b32),(b32-b42),1)\
        +tf.tensordot(tf.transpose(lambda_b33),(b33-b43),1)

        lambdas = lambdas_w + lambdas_b
        loss = cross_entropy + reg + lambdas
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        
        train_W1 = optimizer.minimize(loss, var_list=[W11,W12, W13, b11, b12, b13])
        train_W3 = optimizer.minimize(loss, var_list=[W31,W32,W33,b31,b32,b33])
        train_W2 = optimizer.minimize(loss, var_list=[W21,W22,W23,b21,b22,b23])
        train_W4 = optimizer.minimize(loss, var_list=[W41,W42,W43, b41,b42,b43])
        
end = time.time()
print('Time in seconds to run the model: ' ,end-start)
gc.collect()

max_acc = max(avgAcc_Test)
print(avgAcc_Test[-1])
print('The maximum training accuracy is: ',max(avgAcc_Train))
print('The maximum test accuracy is: ',max_acc)
print(len(avgAcc_Test))
print('Iterations taken to reach the max accuracy is: ', avgAcc_Test.index(max_acc))



