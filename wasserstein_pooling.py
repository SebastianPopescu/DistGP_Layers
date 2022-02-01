# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
DTYPE=tf.float32

###########################
##### helper functions ####
###########################

def get_feature_blocks(X, dim_filter, num_stride):

    ##### X -- features shape (num_batches, height, width, num_filters)
    ### get sizes 
    shapes = X.get_shape().as_list()
    num_batches = shapes[0]
    height = shapes[1]
    width = shapes[2]
    num_filters = shapes[3]
    
    patches = tf.extract_image_patches(images = X, ksizes = [1, dim_filter, dim_filter, 1], 
        strides = [1, num_stride, num_stride, 1], 
        rates = [1, 1, 1, 1],  padding = "VALID")

    new_shape = patches.get_shape().as_list()
    patches = tf.reshape(patches, [-1,new_shape[1], new_shape[2], dim_filter**2, num_filters])

    ### patches -- shape (num_batch,new_height, new_width,  dim_filter**2, num_channels)
    
    return patches

def get_3D_feature_blocks(X, dim_filter, num_stride):

    #### X -- shape (num_batch, height, width, depth, num_channels)
    ### get sizes ###
    lista_shapes = X.get_shape().as_list()
    lista_shape = tf.shape(X)
    new_height = lista_shapes[1]
    new_height = (new_height - dim_filter) // num_stride +1     
    new_width = lista_shapes[2]
    new_width = (new_width - dim_filter) // num_stride +1   
    new_depth = lista_shapes[3]
    new_depth = (new_depth - dim_filter) // num_stride +1   

    num_channels = lista_shapes[-1]
    X = tf.reshape(X, [-1, lista_shapes[2], lista_shapes[3], lista_shapes[4]])

    patches = tf.extract_image_patches(images = X, ksizes = [1, dim_filter, dim_filter, 1], 
        strides = [1, num_stride, num_stride, 1], rates = [1, 1, 1, 1], padding = "VALID")
    ## patches -- shape (num_batch*height, new_width, new_depth, num_channels * dim_filter**3)
    patches = tf.reshape(patches, [-1, lista_shapes[1], new_width * new_depth, num_channels * dim_filter**2])

    patches = tf.extract_image_patches(images = patches, ksizes = [1, dim_filter, 1, 1], 
        strides = [1, num_stride, 1, 1], rates = [1, 1, 1, 1], padding = "VALID")

    ### patches -- shape (num_batches, new_height, new_width* new_depth, dim_filter**3 * num_channels)    
    
    patches =  tf.cast(tf.reshape(patches, [-1, new_height, new_width, new_depth, dim_filter**3, num_channels]), DTYPE)
    ### patches -- shape (num_batches * new_height * new_width, dim_filter**2 * depth)
    
    return patches

def wasserstein_spatial_pooling(l, input_mean, input_var, dim_layer, dim_filter, num_stride, data_type, full_cov=False):

    ###########################################################################
    ### input_mean -- shape (num_batch, height, width, depth, num_channels) ###	
    ### input_var -- shape (num_batch, height, width, depth, num_channels) ###	    
    ###########################################################################

    #### get blocks from the feature space of the previous layer ####
    print('**************************************************')
    print('** we are inside propagate function '+str(l)+' ***')

    if data_type == '2D':

        mean_patches = get_feature_blocks(X = input_mean, dim_filter = dim_filter, num_stride = num_stride)
        var_patches = get_feature_blocks(X = input_var, dim_filter = dim_filter, num_stride = num_stride)       
        ### mean_patches -- shape (num_batch, new_height, new_width,  dim_filter**2, num_channels)

    elif data_type == '3D':

        mean_patches = get_3D_feature_blocks(X = input_mean, dim_filter = dim_filter, num_stride = num_stride)
        var_patches = get_3D_feature_blocks(X = var_mean, dim_filter = dim_filter, num_stride = num_stride)        
        ### mean_patches -- shape (num_batch, new_height, new_width,  depth, dim_filter**3, num_channels)
        
    print('size of patches')
    print(mean_patches)
    print(var_patches)

    ###############################################
    ### perform the Wasserstein spatial pooling ###
    ###############################################

    new_shape = mean_patches.get_shape().as_list()
    mean_patches = tf.reshape(mean_patches, [-1, new_shape[-1]])
    var_patches = tf.reshape(var_patches, [-1, new_shape[-1]])

    #############################################
    ### construct the barycentric coordinates ###
    #############################################
    kkt_pe_bat = mean_patches.get_shape().as_list()

    if data_type=='2D':

        barycentric_coordinates = tf.ones((kkt_pe_bat[0],1)) * (1.0/(dim_filter*dim_filter))
    
    elif data_type=='3D':

        barycentric_coordinates = tf.ones((kkt_pe_bat[0],1)) * (1.0/(dim_filter*dim_filter*dim_filter))

    wass_bary_mean, wass_bary_var = wasserstein_barycentre_gaussian_measures(gaussian_means = mean_patches, gaussian_vars = var_patches, 
        barycentric_coordinates = barycentric_coordinates, num_iterations = 10)
    ### wass_bary_mean -- shape (num_batches, 1)
    ### wass_bary_var -- shape (num_batches, 1)

    if data_type == '2D':

        wass_bary_mean = tf.reshape(wass_bary_mean, [-1, new_shape[1], new_shape[2], new_shape[-1])
        wass_bary_var = tf.reshape(wass_bary_mean, [-1,  new_shape[1], new_shape[2], new_shape[-1])    
    
    elif data_type == '3D':

        wass_bary_mean = tf.reshape(wass_bary_mean, [-1, new_shape[1], new_shape[2], new_shape[3], new_shape[-1])
        wass_bary_var = tf.reshape(wass_bary_mean, [-1,  new_shape[1], new_shape[2], new_shape[3], new_shape[-1])    

    print('***** end of Wasserstein spatial pooling layer  '+str(l)+' ***')
    print(wass_bary_mean)
    print(wass_bary_var)

    return wass_bary_mean, wass_bary_var