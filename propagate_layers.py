# -*- coding: utf-8 -*-
import numpy as np
from model_details import model_details
from kullback_lieblers import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
DTYPE=tf.float32

#### Diagnostic function ####

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

###########################################################
##### propagate functions for CNNs -- operating on 2D #####
###########################################################

def get_feature_blocks(X, dim_filter, num_stride):

    ##### X -- features shape (num_batches, height, width, depth, num_filters)
    ### get sizes 
    shapes = X.get_shape().as_list()
    num_batches = shapes[0]
    height = shapes[1]
    width = shapes[2]
    depth = shapes[3]
    
    patches = tf.extract_image_patches(images = X, ksizes = [1, dim_filter, dim_filter, 1], 
        strides = [1, num_stride, num_stride, 1], 
        rates = [1, 1, 1, 1],  padding = "VALID")

    return patches

###########################################################
##### propagate functions for CNNs -- operating on 3D #####
###########################################################

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
    ## patches -- shape (num_batch*height, new_width, new_depth, num_channels * dim_filter**2)
    patches = tf.reshape(patches, [-1, lista_shapes[1], new_width * new_depth, num_channels * dim_filter**2])

    patches = tf.extract_image_patches(images = patches, ksizes = [1, dim_filter, 1, 1], 
        strides = [1, num_stride, 1, 1], rates = [1, 1, 1, 1], padding = "VALID")

    ### patches -- shape (num_batches, new_height, new_width* new_depth, dim_filter**3 * num_channels)    
    
    patches =  tf.cast(tf.reshape(patches, [-1, new_height, new_width, new_depth, dim_filter**3 * num_channels]), DTYPE)
    ### patches -- shape (num_batches * new_height * new_width, dim_filter**2 * depth)
    
    return patches


class propagate_layers(model_details):

    def __init__(self, **kwargs):

        model_details.__init__(self, **kwargs)

    def propagate_euclidean_CNN(self, l, input_prev_layer, conditional_GP_object, layer_type, training_time, full_cov=False):

        ##### this operates just on the first layer #####
        ##### standard GP-based mapping between input and first hidden layer ###
        ##### as in DeepConvGP paper ####

        ###########################################################################
        ### input_prev -- shape (num_batch, height, width, depth, num_channels) ###	
        ###########################################################################

        #### get blocks from the feature space of the previous layer ####
        print('**************************************************')
        print('** we are inside propagate function '+str(l)+' ***')

        if self.data_type == '2D':
            patches = get_feature_blocks(X = input_prev_layer, dim_filter = self.dim_filters[l-1], num_stride = self.num_strides[l-1])
        elif self.data_type == '3D':
            patches = get_3D_feature_blocks(X = input_prev_layer, dim_filter = self.dim_filters[l-1], num_stride = self.num_strides[l-1])

        print('size of patches')
        print(patches)
        
        num_input_channels = input_prev_layer.get_shape().as_list()[-1]
        new_shape = patches.get_shape().as_list()
        new_height = new_shape[1]
        new_width = new_shape[2]
        new_num_input_channels = new_shape[-1]    
        
        if self.data_type=='3D':
            new_depth = new_shape[3]

        patches =  tf.cast(tf.reshape(patches, [-1, new_num_input_channels]), DTYPE)        
        with tf.variable_scope('layer_'+str(l), reuse = tf.AUTO_REUSE):

            if training_time:
                Z = tf.get_variable(initializer =  tf.constant(self.Z_init),
                    dtype = DTYPE, name = 'Z')	
            else:
                if self.data_type=='2D':
                    Z = tf.get_variable(initializer = tf.zeros_initializer(), shape=(self.num_inducing[l-1], self.dim_filters[l-1]**2*self.dim_layers[l-1])  ,name = 'Z', dtype = DTYPE)	
                elif self.data_type=='3D':
                    Z = tf.get_variable(initializer = tf.zeros_initializer(), shape=(self.num_inducing[l-1], self.dim_filters[l-1]**3*self.dim_layers[l-1])  ,name = 'Z', dtype = DTYPE)

            if self.data_type=='2D':
                log_lengthscales = tf.get_variable(initializer = tf.constant([2.0 for _ in range(self.dim_filters[l-1]**2*self.dim_layers[l-1])],
                    dtype=DTYPE), dtype=DTYPE,name='log_lengthscales')
            elif self.data_type=='3D':
                log_lengthscales = tf.get_variable(initializer = tf.constant([2.0 for _ in range(self.dim_filters[l-1]**3*self.dim_layers[l-1])],
                    dtype=DTYPE), dtype=DTYPE,name='log_lengthscales')

            log_kernel_variance = tf.get_variable(initializer = tf.constant(-0.301, dtype=DTYPE),dtype=DTYPE,
                name='log_kernel_variance')
            if training_time:
                variable_summaries(var = Z, name = 'Z')
                variable_summaries(var = log_lengthscales, name = 'log_Lengthscales')
                variable_summaries(var = log_kernel_variance, name = 'log_kernel_variance')


            q_mu = tf.get_variable(initializer = tf.zeros_initializer(), shape=(self.num_inducing[l-1], self.dim_layers[l]),
                dtype = DTYPE, name='q_mu')


            matrice_identica = np.eye(self.num_inducing[l-1], dtype = np.float32)  * 1e-2
            tiled_matrice_identica = np.tile(np.expand_dims(matrice_identica,axis=0),(self.dim_layers[l],1,1))
            q_sqrt_var = tf.get_variable(initializer = tf.constant(tiled_matrice_identica, dtype=tf.float32),
                dtype=DTYPE, name='q_sqrt_var')
            q_sqrt_var = tf.matrix_band_part(q_sqrt_var,-1,0)
            #q_sqrt = tf.tile(tf.expand_dims(q_sqrt_var,axis=0),[self.dim_layers[l], 1, 1])

            if training_time:               

                #kl_cost = KL(q_mu,q_sqrt))
                kl_cost = gauss_kl(q_mu = q_mu, q_sqrt = q_sqrt_var)


            if training_time:
                variable_summaries(var = q_mu, name = 'q_mu')
                variable_summaries(var = q_sqrt_var, name = 'q_sqrt_var')

        output_mean, output_var_epistemic, output_var_distributional = conditional_GP_object.conditional_euclidean(Xnew = patches, X = Z, l = l, 
            q_mu = q_mu,
            q_sqrt = q_sqrt_var,
            log_lengthscales = log_lengthscales,
            log_kernel_variance = log_kernel_variance,
            layer_type = layer_type,
            full_cov = full_cov)


        #### output_mean -- shape (num_batches * new_height * new_width * new_depth if 3D data, dim_layer)		
        #### output_var -- shape (num_batches * new_height * new_width * new_depth if 3D data, dim_layer)		
        if self.data_type == '2D':
            
            output_mean = tf.reshape(output_mean, [-1, new_height, new_width, self.dim_layers[l]])
            output_var_epistemic = tf.reshape(output_var_epistemic, [-1, new_height, new_width, self.dim_layers[l]])    
            output_var_distributional =  tf.reshape(output_var_distributional, [-1, new_height, new_width, self.dim_layers[l]])    

        elif self.data_type == '3D':

            output_mean = tf.reshape(output_mean, [-1, new_height, new_width, new_depth, self.dim_layers[l]])
            output_var_epistemic = tf.reshape(output_var_epistemic, [-1, new_height, new_width, new_depth, self.dim_layers[l]])    
            output_var_distributional = tf.reshape(output_var_distributional, [-1, new_height, new_width, new_depth, self.dim_layers[l]])  

        print('***** end of layer '+str(l)+' ***')
        print(output_mean)
        print(output_var_epistemic)
        print(output_var_distributional)

        if training_time:
            return output_mean, output_var_epistemic, output_var_distributional, kl_cost
        else:
            return output_mean, output_var_epistemic, output_var_distributional

    def propagate_wasserstein_CNN(self, l, X_mean, X_var, conditional_GP_object, layer_type, training_time, full_cov=False):

        new_shape = X_mean.get_shape().as_list()    
        new_num_input_channels = new_shape[-1]

        ##### used for embeddings between hidden layers if l!=self.num_layers
        ##### apply standard convolutional approach on mean and var ####
        ##### use Distributional GP as an activation function ##########

        #### if l==self.num_layers, just does a one-to-one mapping to output space ###


        #############################################################################################
        ### X_mean -- shape (num_batch, height, width, depth, num_channels) if 3D data ##############	
        ### X_var -- shape (num_batch, height, width, depth, num_channels) if 3D data ###############	
        #############################################################################################

        print('**************************************************')
        print('** we are inside propagate function '+str(l)+' ***')
        print('**************************************************')


        #########################
        ### get filter weight ###
        #########################

        with tf.variable_scope('layer_'+str(l), reuse = tf.AUTO_REUSE):
            #### initalize filter for the convolutionally warped GP part ####
            if self.data_type=='2D':
                current_filter = tf.get_variable(initializer = tf.initializers.variance_scaling(), 
                    shape =  (self.dim_filters[l-1], self.dim_filters[l-1], self.dim_layers[l-1], self.dim_convolutionally_warped_gp[l-1]),
                    name='filters', dtype = DTYPE)
            elif self.data_type=='3D':
                current_filter = tf.get_variable(initializer = tf.initializers.variance_scaling(), 
                    shape =  (self.dim_filters[l-1], self.dim_filters[l-1], self.dim_filters[l-1], self.dim_layers[l-1], self.dim_convolutionally_warped_gp[l-1]),
                    name='filters', dtype = DTYPE)

        ###########################
        ##### Mean Convolution ####
        ###########################

        if self.data_type=='2D':
            if l==2:
                dilations = [1, 2, 2, 1]
            else:
                dilations = [1, 1, 1, 1]

            X_mean  = tf.nn.conv2d(
                input = X_mean,
                filter = current_filter,
                strides=(1, self.num_strides[l-1], self.num_strides[l-1], 1),
                padding='VALID',
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                dilations=dilations,
                name='mean_conv_2d')
        
        elif self.data_type=='3D':
            if l==2:
                dilations = [1, 2, 2, 2, 1]
            else:
                dilations = [1, 1, 1, 1, 1]

            X_mean  = tf.nn.conv3d(
                input = X_mean,
                filter = current_filter,
                strides=(1, self.num_strides[l-1], self.num_strides[l-1], self.num_strides[l-1], 1),
                padding='VALID',
                dilations=dilations,
                name='mean_conv_3d')

        ###########################
        ##### Var Convolution #####
        ###########################

        if self.data_type == '2D':
            if l==2:
                dilations = [1, 2, 2, 1]
            else:
                dilations = [1, 1, 1, 1]

            X_var  = tf.nn.conv2d(
                input = X_var,
                filter = tf.square(current_filter),
                strides=(1, self.num_strides[l-1], self.num_strides[l-1], 1),
                padding='VALID',
                use_cudnn_on_gpu=True,
                data_format='NHWC',
                dilations=dilations,
                name='var_conv_2d')

        elif self.data_type == '3D':
            if l==2:
               dilations =  [1, 2, 2, 2, 1]
            else:
                dilations = [1, 1, 1, 1, 1]
            X_var  = tf.nn.conv3d(
                input = X_var,
                filter = tf.square(current_filter),
                strides=(1, self.num_strides[l-1], self.num_strides[l-1], self.num_strides[l-1], 1),
                padding='VALID',
                dilations=dilations,
                name='var_conv_3d')


        new_shape = X_mean.get_shape().as_list()
        new_height = new_shape[1]
        new_width = new_shape[2]
        new_num_input_channels = new_shape[-1]
        if self.data_type=='3D':
            new_depth = new_shape[3]

        X_mean =  tf.cast(tf.reshape(X_mean, [-1, new_num_input_channels]), DTYPE)
        X_var =  tf.cast(tf.reshape(X_var, [-1, new_num_input_channels]), DTYPE)        

        with tf.variable_scope('layer_'+str(l), reuse = tf.AUTO_REUSE):

            Z_mean = tf.get_variable(initializer =  tf.random_uniform_initializer(minval=-0.25, maxval=1.5),
                shape = (self.num_inducing[l-1], self.dim_convolutionally_warped_gp[l-1]),
                dtype = DTYPE, name = 'Z_mean')

            ### variance of inducing points location is taken to be diagonal multivariate #####
            matrice_dummy = tf.ones(shape = [self.num_inducing[l-1], self.dim_convolutionally_warped_gp[l-1]], dtype=DTYPE) * 0.25
            Z_var_sqrt = tf.get_variable(initializer = matrice_dummy,
                dtype=DTYPE,name='Z_var_sqrt')								
            Z_var = tf.square(Z_var_sqrt)

            log_lengthscales = tf.get_variable(initializer = tf.constant([2.0 for _ in range(self.dim_convolutionally_warped_gp[l-1])],
                dtype = DTYPE), dtype = DTYPE, name = 'log_lengthscales')

            log_kernel_variance = tf.get_variable(initializer = tf.constant(-0.301, dtype = DTYPE), dtype = DTYPE,
                name = 'log_kernel_variance')

            if training_time:
                 
                variable_summaries(var = Z_mean, name = 'Z_mean')
                variable_summaries(var = Z_var, name = 'Z_var')
                variable_summaries(var = log_lengthscales, name = 'log_lengthscales')
                variable_summaries(var = log_kernel_variance, name=  'log_kernel_variance')
  
            q_mu = tf.get_variable(initializer = tf.zeros_initializer(),
                shape=(self.num_inducing[l-1], self.dim_layers[l]), dtype = DTYPE, name = 'q_mu')


            matrice_identica = np.eye(self.num_inducing[l-1], dtype = np.float32)  
            if l != self.num_layers:
                matrice_identica = matrice_identica * 1e-2
            tiled_matrice_identica = np.tile(np.expand_dims(matrice_identica,axis=0),(self.dim_layers[l],1,1))
            q_sqrt_var = tf.get_variable(initializer = tf.constant(tiled_matrice_identica, dtype=tf.float32),
                dtype=DTYPE, name='q_sqrt_var')
            q_sqrt_var = tf.matrix_band_part(q_sqrt_var,-1,0)
            #q_sqrt = tf.tile(tf.expand_dims(q_sqrt_var,axis=0),[self.dim_layers[l], 1, 1])

            if training_time:               

                #kl_cost = KL(q_mu,q_sqrt))
                kl_cost = gauss_kl(q_mu = q_mu, q_sqrt = q_sqrt_var)


            if training_time:
                variable_summaries(var = q_mu, name = 'q_mu')
                variable_summaries(var = q_sqrt_var, name = 'q_sqrt_var')

        output_mean, output_var_epistemic, output_var_distributional = conditional_GP_object.conditional_wasserstein(Xnew_mean = X_mean,
            X_mean = Z_mean, Xnew_var = X_var, X_var = Z_var, 
            l = l, 
            q_mu = q_mu,
            q_sqrt = q_sqrt_var,
            log_lengthscales = log_lengthscales,
            log_kernel_variance =  log_kernel_variance,
            layer_type = layer_type)

        output_mean = output_mean + tf.tile(tf.reduce_mean(X_mean, axis=-1, keepdims=True),[1,self.dim_layers[l]])

        #### output_mean -- shape (num_batches * new_height * new_width * new_depth if 3D data, dim_layer)		
        #### output_var -- shape (num_batches * new_height * new_width * new_depth if 3D data, dim_layer)		

        if self.data_type == '2D':
            output_mean = tf.reshape(output_mean, [-1, new_height, new_width, self.dim_layers[l]])
            output_var_epistemic = tf.reshape(output_var_epistemic, [-1, new_height, new_width, self.dim_layers[l]])    
            output_var_distributional = tf.reshape(output_var_distributional, [-1, new_height, new_width, self.dim_layers[l]])    

        elif self.data_type == '3D':
            output_mean = tf.reshape(output_mean, [-1, new_height, new_width, new_depth, self.dim_layers[l]])
            output_var_epistemic = tf.reshape(output_var_epistemic, [-1, new_height, new_width, new_depth, self.dim_layers[l]])            
            output_var_distributional = tf.reshape(output_var_distributional, [-1, new_height, new_width, new_depth, self.dim_layers[l]])     

        print('***** end of layer '+str(l)+' ***')
        print(output_mean)
        print(output_var_epistemic)
        print(output_var_distributional)

        if training_time:
            return output_mean, output_var_epistemic, output_var_distributional, kl_cost 
        else:
            return output_mean, output_var_epistemic, output_var_distributional


    def propagate_wasserstein_CNN_last_layer(self, l, X_mean, X_var, conditional_GP_object, layer_type, training_time, full_cov=False):

        new_shape = X_mean.get_shape().as_list()    
        new_num_input_channels = new_shape[-1]

        ##### used for embeddings between hidden layers if l!=self.num_layers
        ##### apply standard convolutional approach on mean and var ####
        ##### use Distributional GP as an activation function ##########

        #### if l==self.num_layers, just does a one-to-one mapping to output space ###


        #############################################################################################
        ### X_mean -- shape (num_batch, height, width, depth, num_channels) if 3D data ##############	
        ### X_var -- shape (num_batch, height, width, depth, num_channels) if 3D data ###############	
        #############################################################################################

        print('**************************************************')
        print('** we are inside propagate function '+str(l)+' ***')
        print('**************************************************')

        new_shape = X_mean.get_shape().as_list()
        new_height = new_shape[1]
        new_width = new_shape[2]
        new_num_input_channels = new_shape[-1]
        if self.data_type=='3D':
            new_depth = new_shape[3]

        X_mean =  tf.cast(tf.reshape(X_mean, [-1, new_num_input_channels]), DTYPE)
        X_var =  tf.cast(tf.reshape(X_var, [-1, new_num_input_channels]), DTYPE)        

        with tf.variable_scope('layer_'+str(l), reuse = tf.AUTO_REUSE):

            Z_mean = tf.get_variable(initializer =  tf.random_uniform_initializer(minval=-0.25, maxval=1.5),
                shape = (self.num_inducing[l-1], self.dim_layers[l-1]),
                dtype = DTYPE, name = 'Z_mean')

            ### variance of inducing points location is taken to be diagonal multivariate #####
            matrice_dummy = tf.ones(shape = [self.num_inducing[l-1], self.dim_layers[l-1]], dtype=DTYPE) * 0.25
            Z_var_sqrt = tf.get_variable(initializer = matrice_dummy,
                dtype=DTYPE,name='Z_var_sqrt')								
            Z_var = tf.square(Z_var_sqrt)

            log_lengthscales = tf.get_variable(initializer = tf.constant([2.0 for _ in range(self.dim_layers[l-1])],
                dtype = DTYPE), dtype = DTYPE, name = 'log_lengthscales')

            log_kernel_variance = tf.get_variable(initializer = tf.constant(-0.301, dtype = DTYPE), dtype = DTYPE,
                name = 'log_kernel_variance')
            
            if training_time:     
                variable_summaries(var = Z_mean, name = 'Z_mean')
                variable_summaries(var = Z_var, name = 'Z_var')
                variable_summaries(var = log_lengthscales, name = 'log_lengthscales')
                variable_summaries(var = log_kernel_variance, name=  'log_kernel_variance')
                            
            q_mu = tf.get_variable(initializer = tf.zeros_initializer(),
                shape=(self.num_inducing[l-1], self.dim_layers[l]), dtype = DTYPE, name = 'q_mu')

            matrice_identica = np.eye(self.num_inducing[l-1], dtype = np.float32)  
            if l != self.num_layers:
                matrice_identica = matrice_identica * 1e-2
            tiled_matrice_identica = np.tile(np.expand_dims(matrice_identica,axis=0),(self.dim_layers[l],1,1))
            q_sqrt_var = tf.get_variable(initializer = tf.constant(tiled_matrice_identica, dtype=tf.float32),
                dtype=DTYPE, name='q_sqrt_var')
            q_sqrt_var = tf.matrix_band_part(q_sqrt_var,-1,0)
            #q_sqrt = tf.tile(tf.expand_dims(q_sqrt_var,axis=0),[self.dim_layers[l], 1, 1])

            if training_time:               

                #kl_cost = KL(q_mu,q_sqrt))
                kl_cost = gauss_kl(q_mu = q_mu, q_sqrt = q_sqrt_var)


            if training_time:
                variable_summaries(var = q_mu, name = 'q_mu')
                variable_summaries(var = q_sqrt_var, name = 'q_sqrt_var')

        output_mean, output_var_epistemic, output_var_distributional = conditional_GP_object.conditional_wasserstein(Xnew_mean = X_mean,
            X_mean = Z_mean, Xnew_var = X_var, X_var = Z_var, 
            l = l, 
            q_mu = q_mu,
            q_sqrt = q_sqrt_var,
            log_lengthscales = log_lengthscales,
            log_kernel_variance =  log_kernel_variance,
            layer_type = layer_type)
        
        #### output_mean -- shape (num_batches * new_height * new_width * new_depth if 3D data, dim_layer)		
        #### output_var -- shape (num_batches * new_height * new_width * new_depth if 3D data, dim_layer)		

        if self.data_type == '2D':
            output_mean = tf.reshape(output_mean, [-1, new_height, new_width, self.dim_layers[l]])
            output_var_epistemic = tf.reshape(output_var_epistemic, [-1, new_height, new_width, self.dim_layers[l]])    
            output_var_distributional = tf.reshape(output_var_distributional, [-1, new_height, new_width, self.dim_layers[l]])    

        elif self.data_type == '3D':
            output_mean = tf.reshape(output_mean, [-1, new_height, new_width, new_depth, self.dim_layers[l]])
            output_var_epistemic = tf.reshape(output_var_epistemic, [-1, new_height, new_width, new_depth, self.dim_layers[l]])            
            output_var_distributional = tf.reshape(output_var_distributional, [-1, new_height, new_width, new_depth, self.dim_layers[l]])     

        print('***** end of layer '+str(l)+' ***')
        print(output_mean)
        print(output_var_epistemic)
        print(output_var_distributional)

        if training_time:
            return output_mean, output_var_epistemic, output_var_distributional, kl_cost
        else:
            return output_mean, output_var_epistemic, output_var_distributional

























### Useful for U-NET ###  

### TODO -- add function that upsamples

### TODO -- add function that crops the centre and copies it 
