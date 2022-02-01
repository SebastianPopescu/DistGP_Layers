# -*- coding: utf-8 -*-
import numpy as np
from wasserstein_kernels import *
from model_details import model_details
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
DTYPE=tf.float32

def eye(N):
    
    return tf.diag(tf.ones(tf.stack([N,]),dtype=DTYPE))

def condition(X):

    return X + eye(tf.shape(X)[0]) * 1e-3


class conditional_GP(model_details):

    def __init__(self, **kwargs):

        model_details.__init__(self, **kwargs)

    def conditional_euclidean(self, Xnew, X, l, 
        q_mu, q_sqrt, log_lengthscales, log_kernel_variance,
        layer_type,
        white = True, full_cov = False):

        ### layer_type = 'DNN' or 'CNN'

        type_var='full'

        Kmm = RBF(X, X, log_lengthscales, log_kernel_variance)
        Kmm = condition(Kmm)
        Kmn = RBF(X, Xnew, log_lengthscales, log_kernel_variance)
        
        if full_cov:
            Knn = RBF(Xnew,Xnew,log_lengthscales,log_kernel_variance)
        else:
            Knn = RBF_Kdiag(Xnew,log_kernel_variance)
        
        Lm = tf.cholesky(Kmm)
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)
        
        if full_cov:
            fvar_distributional = Knn - tf.matmul(A, A, transpose_a=True)
            ### fvar_distributional -- shape (num_batch, num_batch)
            ### Warning -- need to expand_dims and then tile to equal dim_layers[l]
            ### Doens't reallty matter since we don't use the option
        else:
            fvar_distributional = Knn - tf.transpose(tf.reduce_sum(tf.square(A), 0,keep_dims=True))
            #### fvar_distributional -- shape (num_batch,1)
            fvar_distributional = tf.tile(fvar_distributional,[1,self.dim_layers[l]])
       
        if not white:
            A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

        fmean = tf.matmul(A, q_mu, transpose_a=True)

        ### get the identity convolutional filter, same approach as in other papers

        if layer_type=='CNN':

            if self.data_type=='2D':

                identity_filter = np.zeros((self.dim_filters[l-1], self.dim_filters[l-1], self.dim_layers[l-1], 1), dtype = np.float32)
                identity_filter[self.dim_filters[l-1] // 2, self.dim_filters[l-1] // 2, :, :] = 1.0
            
            elif self.data_type=='3D':
                
                identity_filter = np.zeros((self.dim_filters[l-1], self.dim_filters[l-1], self.dim_filters[l-1], self.dim_layers[l-1], 1), dtype = np.float32)
                identity_filter[self.dim_filters[l-1] // 2, self.dim_filters[l-1] // 2, self.dim_filters[l-1] // 2, :, :] = 1.0

            identity_filter = np.reshape(identity_filter, (-1,1))
            fmean = fmean + tf.tile(tf.matmul(Xnew, identity_filter),[1,self.dim_layers[l]])


        A_tiled = tf.tile(A[None, ...], [self.dim_layers[l], 1, 1])

        if full_cov:
            LTA= tf.matmul(tf.transpose(q_sqrt,[0,2,1]),A_tiled)
            fvar_epistemic = tf.matmul(LTA,LTA,transpose_a=True)
        else:
            LTA= tf.matmul(tf.transpose(q_sqrt,[0,2,1]),A_tiled)
            fvar_epistemic = tf.reduce_sum(tf.square(LTA),1,keep_dims = False)
            #### fvar_epistemic -- shape (dim_layers, num_batch)
            fvar_epistemic = tf.transpose(fvar_epistemic)

        return fmean, fvar_epistemic, fvar_distributional

    def conditional_wasserstein(self, Xnew_mean, X_mean, Xnew_var, X_var, 
        l,
        q_mu, q_sqrt, log_lengthscales, log_kernel_variance,
        layer_type,    
        white = True, full_cov = False):

        ### layer_type = 'DNN' or 'CNN'

        type_var='full'
        
        Kmm = wasserstein_2_distance_gaussian_kernel(X1_mean = X_mean, X2_mean = X_mean,
            X1_var = X_var, X2_var = X_var, log_lengthscale = log_lengthscales,
            log_kernel_variance = log_kernel_variance)
        Kmm = condition(Kmm)
        Kmn =  wasserstein_2_distance_gaussian_kernel(X1_mean = X_mean, X2_mean = Xnew_mean,
            X1_var = X_var, X2_var = Xnew_var, log_lengthscale = log_lengthscales,
            log_kernel_variance = log_kernel_variance)
        
        if full_cov:
            Knn = RBF(Xnew_mean, Xnew_mean, log_lengthscales, log_kernel_variance)
        else:
            Knn = RBF_Kdiag(Xnew_mean, log_kernel_variance)
        
        Lm = tf.cholesky(Kmm)
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)
        
        if full_cov:
            fvar_distributional = Knn - tf.matmul(A, A, transpose_a=True)
            ### Warning -- doesn't work ###
        else:
            fvar_distributional = Knn - tf.transpose(tf.reduce_sum(tf.square(A), 0,keep_dims=True))
            fvar_distributional = tf.tile(fvar_distributional,[1,self.dim_layers[l]])
        
        if not white:
            A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

        fmean = tf.matmul(A, q_mu, transpose_a=True)

        A_tiled = tf.tile(A[None, ...], [self.dim_layers[l], 1, 1])

        if full_cov:
            LTA= tf.matmul(tf.transpose(q_sqrt,[0,2,1]),A_tiled)
            fvar_epistemic = tf.matmul(LTA,LTA,transpose_a=True)
        else:
            LTA= tf.matmul(tf.transpose(q_sqrt,[0,2,1]),A_tiled)
            fvar_epistemic = tf.reduce_sum(tf.square(LTA),1,keep_dims = False)
            #### fvar_epistemic -- shape (dim_layers, num_batch)
            fvar_epistemic = tf.transpose(fvar_epistemic)

        return fmean, fvar_epistemic, fvar_distributional

