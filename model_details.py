# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import sys
import time
import argparse
import os
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
DTYPE=tf.float32

class model_details(object):

    def __init__(self, num_data, num_test,
        dim_input, dim_output, num_iterations, num_layers, dim_layers, dim_convolutionally_warped_gp,
        num_inducing, type_var, 
        num_batch, num_samples_testing, 
        dim_filters, num_strides, 
        name_dataset, import_model, iteration_restored, learning_rate,
        dim_input_channels, num_patches_per_subject,
        num_subjects, num_averaged_gradients, use_bayesian_deep_supervision,
        testing_time, data_type, name_dataset_to_import, affine, multi_kernel, Z_init, use_masks):

        self.use_masks = use_masks
        self.Z_init = Z_init
        self.multi_kernel = multi_kernel
        self.affine = affine
        self.name_dataset_to_import = name_dataset_to_import

        self.data_type = data_type ### can be either 2D or 3D
        self.testing_time = testing_time
        self.use_bayesian_deep_supervision = use_bayesian_deep_supervision

        self.num_averaged_gradients = num_averaged_gradients
        self.num_subjects = num_subjects
        self.num_patches_per_subject = num_patches_per_subject
        self.dim_convolutionally_warped_gp = dim_convolutionally_warped_gp
        self.name_dataset = name_dataset
        self.import_model = import_model
        self.iteration_restored = iteration_restored
        self.learning_rate = learning_rate

        self.num_strides = num_strides ### list -- stride for each layer 
        self.dim_filters = dim_filters ### list -- dimension of filter for each layer
        self.num_batch = num_batch 
        self.type_var = type_var
        self.num_samples_testing = num_samples_testing
        self.num_data = num_data
        self.num_test = num_test
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_iterations = num_iterations
        self.num_inducing = num_inducing ### list -- number of inducing points for each layer
        self.num_layers = num_layers
        self.dim_layers = dim_layers
        self.dim_input_channels = dim_input_channels

        
        tf.disable_v2_behavior()

        '''

        #### Warning -- doesn't account for dilations #####
        dim_last_layer = dim_input
        ######## get size of last layer ####
        self.list_dim_last_layer = [] #### to be used for BDS routine 

        for _ in range(num_layers-1):

            print(dim_last_layer)
            dim_last_layer = (dim_last_layer - dim_filters[_]) // num_strides[_] +1 
            self.list_dim_last_layer.append(dim_last_layer)

        self.dim_last_layer = dim_last_layer
        '''