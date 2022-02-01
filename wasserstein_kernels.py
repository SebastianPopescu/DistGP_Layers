import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
DTYPE=tf.float32


#### TODO -- implement Product Probability Kernel ######


def wasserstein_2_distance_gaussian_kernel(X1_mean, X2_mean, X1_var, X2_var, log_lengthscale, log_kernel_variance):
      
    ### X1_mean -- (num_1, num_dim) ###
    ### X2_mean -- (num_2, num_dim) ###
    ### X1_var -- (num_1, num_dim) ###
    ### X2_var -- (num_2, num_dim) ###
    ### log_lengthscale -- shape (dim_input, )

    log_lengthscale = tf.expand_dims(log_lengthscale, axis = 0)
    log_lengthscale = tf.expand_dims(log_lengthscale, axis = -1)
    
    num_1 = tf.shape(X1_mean)[0]
    num_2 = tf.shape(X2_mean)[0]

    log_lengthscale = tf.tile(log_lengthscale, [num_1, 1, num_2])

    X1_mean = tf.expand_dims(X1_mean, axis=-1)
    X2_mean = tf.expand_dims(X2_mean, axis=-1)

    X1_var = tf.expand_dims(X1_var, axis=-1)
    X2_var = tf.expand_dims(X2_var, axis=-1)

    X1_mean = tf.tile(X1_mean, [1, 1, num_2])
    X1_var = tf.tile(X1_var, [1, 1, num_2])

    X2_mean = tf.tile(X2_mean, [1, 1, num_1])
    X2_var = tf.tile(X2_var, [1, 1, num_1])

    X2_mean = tf.transpose(X2_mean, perm=[2, 1, 0])
    X2_var = tf.transpose(X2_var, perm=[2, 1, 0])

    ### X1_mean -- (num_1, num_dim, num_2) ###
    ### X2_mean -- (num_1, num_dim, num_2) ###
    ### X1_var -- (num_1, num_dim, num_2) ###
    ### X2_var -- (num_1, num_dim, num_2) ###

    ### calculate the W-2-squared distance ###

    w2_distance_normed_part = tf.square(X1_mean - X2_mean)
    w2_distance_inside_trace = X1_var + X2_var - 2.0 * tf.sqrt(tf.multiply(X1_var, X2_var))
    w2_distance = w2_distance_normed_part + w2_distance_inside_trace

    return tf.exp(log_kernel_variance) * tf.exp(-tf.reduce_sum(w2_distance/tf.exp(log_lengthscale), axis = 1))

def RBF(X1, X2, log_lengthscales, log_kernel_variance):
           
    X1 = X1 / tf.exp(log_lengthscales)
    X2 = X2 / tf.exp(log_lengthscales)
    X1s = tf.reduce_sum(tf.square(X1),1)
    X2s = tf.reduce_sum(tf.square(X2),1)       

    return tf.exp(log_kernel_variance) * tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1))) /2)      

def RBF_Kdiag(X, log_kernel_variance):
    ### returns a list
    return tf.ones((tf.shape(X)[0],1),dtype=tf.float32) * tf.exp(log_kernel_variance)	
