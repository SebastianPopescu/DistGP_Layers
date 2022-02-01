# -*- coding: utf-8 -*-
import numpy as np
from model_details import model_details
from metrics import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
DTYPE=tf.float32

##########################
#### helper functions ####
##########################

def inv_probit(x):

    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter

def RobustMax(inputul):

    #### Warning -- thiis is proobaly wrong ###

    indices = tf.argmax(inputul,1)

    return tf.one_hot(indices,3, 1.0 - 1e-3,1e-3/2)

def variational_expectations(Y, Fmu, Fvar, log_variance_output):

    pi = np.pi
    pi = tf.cast(pi,tf.float32)
    #num_dims = tf.cast(tf.shape(Y)[0],tf.float64)
    return -0.5 * tf.log(2.0 * pi)  - 0.5 * log_variance_output - 0.5 * (tf.square(Y - Fmu) + Fvar) / tf.exp(log_variance_output)

def bernoulli(x,p):

    p = inv_probit(p)
    return tf.log(tf.where(tf.equal(x,1.0),p,1-p))

def MultiClassLikelihood(inputul, outputul):

    #### this is RobustMax approach ####

    hits = tf.equal(tf.expand_dims(tf.argmax(inputul,1),1),tf.cast(outputul,tf.int64))
    yes = tf.ones(tf.shape(outputul)) - 1e-3
    no = tf.zeros(tf.shape(outputul)) + 1e-3/2

    p = tf.where(hits,yes,no)

    return tf.log(p)

def multiclass_helper(inputul, outputul):

    ###############################################################################
    ##### Zoubin and Kim, 2006 paper on multi-class classification with GPs #######
    ###############################################################################
    softmaxed_input = tf.nn.softmax(inputul)+1e-3

    return tf.log(tf.reduce_sum(tf.multiply(softmaxed_input,outputul),axis=1,keepdims=True))



class loss_functions(model_details):

    def __init__(self, **kwargs):

        model_details.__init__(self, **kwargs)


    def classification(self, outputul, list_inputul_mean, list_inputul_var,
        masks, loss_weights, bds_masks, bds_outputul, use_masks):

        ### list_inputul_mean -- list of mean for each layer -- last layer corresponds to the same scale as the last GP segmentation 
        ### list_inputul_var -- list of variance for each layer -- last layer corresponds to the same scale as the last GP segmentation 
        ### bds_masks and bds_outputul are dictionaries ####
        re_error_final = []
        detailed_re_error_final = []

        scale = tf.cast(self.num_data * 500, tf.float32)
        scale /= tf.cast(tf.shape(list_inputul_mean[-1])[0], tf.float32)  # minibatch size
        full_cov = False

        print('****************************')
        print(' start of loss function ')

        if self.use_bayesian_deep_supervision:

            list_bds_inputul_masked = []
            list_bds_masked_outputul = []
                
            for l in range(1, self.num_layers):

                print('----- layer '+str(l)+' --------')
                input_shape = list_inputul_mean[l-1].get_shape().as_list()
                print('what is going inside bayesian deep supervision')
                print(list_inputul_mean[l-1])
                print(list_inputul_var[l-1])

                ####### 1*1 convolution basically ######

                with tf.variable_scope('deep_supervision',reuse=tf.AUTO_REUSE):

                    if self.data_type =='2D':

                        current_filter = tf.get_variable(initializer =  tf.initializers.variance_scaling(),
                            shape = (1, 1, input_shape[-1], self.dim_output), dtype=DTYPE, name='filter_'+str(l))

                        ###########################
                        ##### Mean Convolution ####
                        ###########################

                        bds_mean_pred  = tf.nn.conv2d(
                            input = list_inputul_mean[l-1],
                            filter = current_filter,
                            strides=(1, 1, 1, 1),
                            padding='VALID',
                            use_cudnn_on_gpu=True,
                            data_format='NHWC',
                            dilations=[1, 1, 1, 1],
                            name='bds_mean_conv_2d_layer_'+str(l)
                        )

                        ###########################
                        ##### Var Convolution #####
                        ###########################

                        bds_var_pred  = tf.nn.conv2d(
                            input = list_inputul_var[l-1],
                            filter = tf.square(current_filter),
                            strides=(1, 1, 1, 1),
                            padding='VALID',
                            use_cudnn_on_gpu=True,
                            data_format='NHWC',
                            dilations=[1, 1, 1, 1],
                            name='bds_var_conv_2d_layer_'+str(l)
                        )

                    elif self.data_type=='3D':

                        current_filter = tf.get_variable(initializer =  tf.initializers.variance_scaling(),
                            shape = (1, 1, 1, input_shape[-1], self.dim_output), dtype=DTYPE, name='filter_'+str(l))

                        ###########################
                        ##### Mean Convolution ####
                        ###########################

                        bds_mean_pred  = tf.nn.conv3d(
                            input = list_inputul_mean[l-1],
                            filter = current_filter,
                            strides=(1, 1, 1, 1, 1),
                            padding='VALID',
                            dilations=[1, 1, 1, 1, 1],
                            name='bds_mean_conv_3d_layer_'+str(l))

                        ###########################
                        ##### Var Convolution #####
                        ###########################

                        bds_var_pred  = tf.nn.conv3d(
                            input = list_inputul_var[l-1],
                            filter = tf.square(current_filter),
                            strides=(1, 1, 1, 1, 1),
                            padding='VALID',
                            dilations=[1, 1, 1, 1, 1],
                            name='bds_var_conv_3d_layer_'+str(l))

                    print('this is after we have applied the weight matrix')
                    print(bds_mean_pred)
                    print(bds_var_pred)

                    ###########################################################################################
                    #### mask the prediction and the output so that we gain gradients just from the Retina ####
                    ###########################################################################################

                    shape_of_input_data = bds_mean_pred.get_shape().as_list()
                    shape_of_label_data = bds_var_pred.get_shape().as_list()

                    if self.dim_output==1:

                        #####################################
                        #### Binary classification case #####
                        #####################################

                        bds_inputul_mean = tf.reshape(bds_mean_pred, [-1, 1])
                        bds_inputul_var = tf.reshape(bds_var_pred, [-1, 1])
                        bds_outputul_current = tf.reshape(bds_outputul[l-1], [-1, 1])
                        if use_masks:

                            bds_masks_current = tf.reshape(bds_masks[l-1], [-1, 1])    
                            bds_inputul_mean_masked = tf.boolean_mask(bds_inputul_mean, bds_masks_current)
                            bds_inputul_var_masked = tf.boolean_mask(bds_inputul_var, bds_masks_current)
                            bds_masked_outputul = tf.boolean_mask(bds_outputul_current, bds_masks_current)
                        
                        else:

                            bds_inputul_mean_masked = bds_inputul_mean
                            bds_inputul_var_masked = bds_inputul_var
                            bds_masked_outputul = bds_outputul_current


                    else:
                

                        ##########################################
                        #### Multi-class classification case #####
                        ##########################################

                        if use_masks:

                            bds_inputul_mean_masked = []
                            bds_inputul_var_masked = []
                            masks = tf.reshape(masks, [-1, 1])
                            bds_masked_outputul = []
                        
                            for current_dim in range(self.dim_output):

                                if self.data_type=='3D':

                                    bds_inputul_mean_current = tf.reshape(tf.slice(bds_mean_pred,[0,0,0,0,current_dim],[-1,-1,-1,-1,1]), [-1, 1])
                                    bds_inputul_var_current = tf.reshape(tf.slice(bds_var_pred,[0,0,0,0,current_dim],[-1,-1,-1,-1,1]), [-1, 1])
                                    bds_outputul_current = tf.reshape(tf.slice(bds_outputul[l-1],[0,0,0,0,current_dim],[-1,-1,-1,-1,1]), [-1, 1])

                                elif self.data_type=='2D':

                                    bds_inputul_mean_current = tf.reshape(tf.slice(bds_mean_pred,[0,0,0,current_dim],[-1,-1,-1,1]), [-1, 1])
                                    bds_inputul_var_current = tf.reshape(tf.slice(bds_var_pred,[0,0,0,current_dim],[-1,-1,-1,1]), [-1, 1])
                                    bds_outputul_current = tf.reshape(tf.slice(bds_outputul[l-1],[0,0,0,current_dim],[-1,-1,-1,1]), [-1, 1])

                            
                                bds_inputul_mean_masked.append(tf.expand_dims(tf.boolean_mask(bds_inputul_mean_current, masks), axis=-1))
                                bds_inputul_var_masked.append(tf.expand_dims(tf.boolean_mask(bds_inputul_var_current, masks),axis=-1))
                                bds_masked_outputul.append(tf.expand_dims(tf.boolean_mask(bds_outputul_current, masks),axis=1))
                        

                            bds_inputul_mean_masked = tf.concat(bds_inputul_mean_masked, axis = -1)
                            bds_inputul_var_masked = tf.concat(bds_inputul_var_masked, axis = -1)
                            bds_masked_outputul = tf.concat(bds_masked_outputul, axis = -1)

                        else:

                        
                            bds_inputul_mean_masked = tf.reshape(bds_mean_pred,[-1,self.dim_output])
                            bds_inputul_var_masked = tf.reshape(bds_var_pred,[-1,self.dim_output])
                            bds_masked_outputul = tf.reshape(bds_outputul[l-1],[-1,self.dim_output])
               
                    print('shapes of tensor in multi-class loss function')
                    print(bds_inputul_mean_masked)
                    print(bds_inputul_var_masked)
                    print(bds_masked_outputul)

                    ##### sample multiple times #####

                    if self.dim_output==1:

                        bds_inputul_mean_masked = tf.expand_dims(bds_inputul_mean_masked, axis = -1)
                        bds_inputul_var_masked = tf.expand_dims(bds_inputul_var_masked, axis = -1)


                    num_samples_testing = 10
                    bds_inputul_masked = tf.tile(tf.expand_dims(bds_inputul_mean_masked,axis=-1), [1,1,num_samples_testing]) + tf.multiply(tf.sqrt(tf.tile(tf.expand_dims(bds_inputul_var_masked,axis=-1),
                        [1,1,num_samples_testing])), tf.random_normal(shape=(tf.shape(bds_inputul_mean_masked)[0],
                        self.dim_output, num_samples_testing), dtype=DTYPE))	

                    bds_inputul_masked = tf.reduce_mean(bds_inputul_masked, axis = -1, keepdims = False)
                    #bds_inputul_masked = tf.reshape(bds_inputul_masked, [-1,self.dim_output])

                    if self.dim_output==1:

                        ##### Binary classification #####
                        bds_inputul_masked = tf.squeeze(bds_inputul_masked)
                        re_error_final.append(loss_weights[l-1] * scale * tf.reduce_sum(bernoulli(p = bds_inputul_masked, x = bds_masked_outputul)))	

                    else:
            
                        ###### Multi-class Classification ######            
                        re_error_final.append(loss_weights[l-1] * scale * tf.reduce_sum(multiclass_helper(inputul = bds_inputul_masked, outputul = bds_masked_outputul)))

                    list_bds_inputul_masked.append(bds_inputul_masked)
                    list_bds_masked_outputul.append(bds_masked_outputul)    
        
        ###########################################################################################
        #### mask the prediction and the output so that we gain gradients just from the Retina ####
        ###########################################################################################

        inputul_mean = list_inputul_mean[-1]
        inputul_var = list_inputul_var[-1]

        shape_of_input_data = inputul_mean.get_shape().as_list()
        shape_of_label_data = outputul.get_shape().as_list()

        if self.dim_output==1:

            #### Binary classification case #####
            inputul_mean = tf.reshape(inputul_mean, [-1, 1])
            inputul_var = tf.reshape(inputul_var, [-1, 1])
            outputul = tf.reshape(outputul, [-1, 1])
            if use_masks:
                masks = tf.reshape(masks, [-1, 1])
                
                inputul_mean_masked = tf.boolean_mask(inputul_mean, masks)
                inputul_var_masked = tf.boolean_mask(inputul_var, masks)
                masked_outputul = tf.boolean_mask(outputul, masks)
            else:
                inputul_mean_masked = inputul_mean
                inputul_var_masked = inputul_var
                masked_outputul = outputul
        
        else:

            ##########################################
            #### Multi-class classification case #####
            ##########################################

            inputul_mean_masked = []
            inputul_var_masked = []
            if use_masks:
                masks = tf.reshape(masks, [-1, 1])
            masked_outputul = []

            if use_masks:

                for current_dim in range(self.dim_output):

                    inputul_mean_current = tf.reshape(inputul_mean[...,current_dim], [-1, 1])
                    inputul_var_current = tf.reshape(inputul_var[...,current_dim], [-1, 1])
                    outputul_current = tf.reshape(outputul[...,current_dim], [-1, 1])

                    inputul_mean_masked.append(tf.expand_dims(tf.boolean_mask(inputul_mean_current, masks), axis=-1))
                    inputul_var_masked.append(tf.expand_dims(tf.boolean_mask(inputul_var_current, masks),axis=-1))
                    masked_outputul.append(tf.expand_dims(tf.boolean_mask(outputul_current, masks),axis=1))


                inputul_mean_masked = tf.concat(inputul_mean_masked, axis = -1)
                inputul_var_masked = tf.concat(inputul_var_masked, axis = -1)
                masked_outputul = tf.concat(masked_outputul, axis = -1)
            else:

                inputul_mean_masked = tf.reshape(inputul_mean,[-1,self.dim_output])
                inputul_var_masked = tf.reshape(inputul_var,[-1,self.dim_output])
                masked_outputul = tf.reshape(outputul,[-1,self.dim_output])

        ###############################
        ##### sample from GP heads ####
        ###############################

        if self.dim_output==1:

            inputul_mean_masked = tf.expand_dims(inputul_mean_masked, axis = -1)
            inputul_var_masked = tf.expand_dims(inputul_var_masked, axis = -1)

        num_samples_testing = 10
        inputul_masked = tf.tile(tf.expand_dims(inputul_mean_masked, axis=-1), [1,1,num_samples_testing]) + tf.multiply(tf.sqrt(tf.tile(tf.expand_dims(inputul_var_masked,axis=-1),
            [1,1,num_samples_testing])), tf.random_normal(shape=(tf.shape(inputul_mean_masked)[0],
            self.dim_output, num_samples_testing), dtype=DTYPE))	

        inputul_masked = tf.reduce_mean(inputul_masked, axis = -1, keepdims = False)
        #inputul_masked = tf.reshape(inputul_masked, [-1,self.dim_output])


        ###############################
        ###### GP loss function #######
        ###############################


        if self.dim_output==1:
            ##### Binary classification #####
            inputul_masked = tf.squeeze(inputul_masked)
            re_error_final.append(loss_weights[-1] * scale * tf.reduce_sum(bernoulli(p = inputul_masked, x = masked_outputul)))

        else:
            ###### Multi-class Classification ######
            re_error_final.append( loss_weights[-1] * scale * tf.reduce_sum(multiclass_helper(inputul = inputul_masked, outputul = masked_outputul)))

        
        re_error_final_aggregated = tf.reduce_sum(re_error_final)

        return re_error_final_aggregated, inputul_masked, masked_outputul, list_bds_inputul_masked, list_bds_masked_outputul, re_error_final

    ### TODO -- need to update this 
    '''
    def regression_error_CNN(inputul, outputul, list_inputul_mean, list_inputul_var,
        dim_output, num_layers, num_data, dim_layers, num_strides, dim_filters, 
        masks, use_bayesian_deep_supervision, loss_weights, bds_masks, bds_outputul,
        data_type):

        inputul_mean = inputul[0]
        inputul_var = inputul[1]

        ### inputul -- list of moments
        ### list_inputul_mean -- list of mean for each layer -- last layer corresponds to the same scale as the last GP segmentation 
        ### list_inputul_var -- list of variance for each layer -- last layer corresponds to the same scale as the last GP segmentation 
        ### bds_masks and bds_outputul are dictionaries ####
        re_error_final = []

        scale = tf.cast(num_data, tf.float32)
        scale /= tf.cast(tf.shape(inputul)[0], tf.float32)  # minibatch size
        full_cov = False

        print('****************************')
        print(' start of loss function ')


        if use_bayesian_deep_supervision:

            list_bds_mean_masked = []
            list_bds_masked_outputul = []
                
            for l in range(1, num_layers):

                input_shape = list_inputul_mean[l-1].get_shape().as_list()
                print('what is going inside bayesian deep supervision')
                print(list_inputul_mean[l-1])

                ####### 1*1 convolution basically ######

                with tf.variable_scope('deep_supervision',reuse=tf.AUTO_REUSE):

                    if data_type =='2D':

                        current_filter = tf.get_variable(initializer =  tf.initializers.variance_scaling(),
                            shape = (1, 1, input_shape[-1], dim_output), dtype=DTYPE, name='filter_'+str(l))

                        ###########################
                        ##### Mean Convolution ####
                        ###########################

                        bds_mean_pred  = tf.nn.conv2d(
                            input = list_inputul_mean[l-1],
                            filter = current_filter,
                            strides=(1, 1, 1, 1),
                            padding='VALID',
                            use_cudnn_on_gpu=True,
                            data_format='NHWC',
                            dilations=[1, 1, 1, 1],
                            name='bds_mean_conv_2d_layer_'+str(l)
                        )

                        ###########################
                        ##### Var Convolution #####
                        ###########################

                        bds_var_pred  = tf.nn.conv2d(
                            input = list_inputul_var[l-1],
                            filter = tf.square(current_filter),
                            strides=(1, 1, 1, 1),
                            padding='VALID',
                            use_cudnn_on_gpu=True,
                            data_format='NHWC',
                            dilations=[1, 1, 1, 1],
                            name='bds_var_conv_2d_layer_'+str(l)
                        )

                    elif data_type=='3D':

                        current_filter = tf.get_variable(initializer =  tf.initializers.variance_scaling(),
                            shape = (1, 1, 1, input_shape[-1], dim_output), dtype=DTYPE, name='filter_'+str(l))

                        ###########################
                        ##### Mean Convolution ####
                        ###########################

                        bds_mean_pred  = tf.nn.conv3d(
                            input = list_inputul_mean[l-1],
                            filter = current_filter,
                            strides=(1, 1, 1, 1, 1),
                            padding='VALID',
                            dilations=[1, 1, 1, 1, 1],
                            name='bds_mean_conv_3d_layer_'+str(l)
                        )

                        ###########################
                        ##### Var Convolution #####
                        ###########################

                        bds_var_pred  = tf.nn.conv3d(
                            input = list_inputul_var[l-1],
                            filter = tf.square(current_filter),
                            strides=(1, 1, 1, 1, 1),
                            padding='VALID',
                            dilations=[1, 1, 1, 1, 1],
                            name='bds_var_conv_3d_layer_'+str(l)
                        )

                    print('this is after we have applied the weight matrix')
                    print(bds_mean_pred)
                    print(bds_var_pred)

                    #noisy_bayesian_deep_supervision = bds_mean_pred + tf.multiply(tf.sqrt(bds_var_pred),
                    #        tf.random_normal(shape=(tf.shape(bds_mean_pred)[0],tf.shape(bds_mean_pred)[1],
                    #        tf.shape(bds_mean_pred)[2],tf.shape(bds_mean_pred)[3]),dtype=tf.float32))
                    #print('literally the prediction')
                    #print(noisy_bayesian_deep_supervision)

                    ###########################################################################################
                    #### mask the prediction and the output so that we gain gradients just from the Retina ####
                    ###########################################################################################

                    shape_of_input_data = bds_mean_pred.get_shape().as_list()
                    shape_of_label_data = bds_var_pred.get_shape().as_list()

                    #####################################
                    #### Binary classification case #####
                    #####################################

                    bds_inputul_mean = tf.reshape(bds_mean_pred, [-1, 1])
                    bds_inputul_var = tf.reshape(bds_var_pred, [-1, 1])
                    bds_outputul_current = tf.reshape(bds_outputul[l-1], [-1, 1])
                    bds_masks_current = tf.reshape(bds_masks[l-1], [-1, 1])
                    
                    bds_inputul_mean_masked = tf.boolean_mask(bds_inputul_mean, bds_masks_current)
                    bds_inputul_var_masked = tf.boolean_mask(bds_inputul_var, bds_masks_current)
                    bds_masked_outputul = tf.boolean_mask(bds_outputul_current, bds_masks_current)

                    print('shapes of tensor in multi-class loss function')
                    print(bds_inputul_mean_masked)
                    print(bds_inputul_var_masked)
                    print(bds_masked_outputul)

                    ###############################
                    ###### GP loss function #######
                    ###############################

                    log_variance_output = tf.get_variable(initializer = tf.constant(-0.301, dtype=DTYPE),dtype=DTYPE,
                        name='log_variance_output_'+str(l))

                    #variable_summaries(log_variance_output, 'log_variance_output_'+str(l))

                    re_error_final.append(loss_weights[l-1] * scale * tf.reduce_sum(variational_expectations(Y = bds_masked_outputul, Fmu = bds_inputul_mean_masked, Fvar = bds_inputul_var_masked, log_variance_output = log_variance_output)))	
                    
                    list_bds_mean_masked.append(bds_inputul_mean_masked)
                    list_bds_masked_outputul.append(bds_masked_outputul)    

        l = num_layers
        inputul_mean, inputul_var = propagate_wasserstein_CNN_last_layer(l = l, X_mean = inputul_mean, X_var = inputul_var,
            dim_layer = dim_layers[l], dim_filter = dim_filters[l-1], num_stride = num_strides[l-1],
            data_type = data_type, full_cov=False)

        ###########################################################################################
        #### mask the prediction and the output so that we gain gradients just from the Retina ####
        ###########################################################################################

        shape_of_input_data = inputul_mean.get_shape().as_list()
        shape_of_label_data = outputul.get_shape().as_list()


        inputul_mean = tf.reshape(inputul_mean, [-1, 1])
        inputul_var = tf.reshape(inputul_var, [-1, 1])
        outputul = tf.reshape(outputul, [-1, 1])
        masks = tf.reshape(masks, [-1, 1])
        
        inputul_mean_masked = tf.boolean_mask(inputul_mean, masks)
        inputul_var_masked = tf.boolean_mask(inputul_var, masks)
        masked_outputul = tf.boolean_mask(outputul, masks)
            

        ###############################
        ###### GP loss function #######
        ###############################


        with tf.variable_scope('layer_'+str(num_layers), reuse = True):
            
            log_variance_output = tf.get_variable('log_variance_output_'+str(l), dtype = DTYPE)

        re_error_final.append(loss_weights[-1] * scale * tf.reduce_sum(variational_expectations(Y = masked_outputul, Fmu = inputul_mean_masked, Fvar = inputul_var_masked, log_variance_output = log_variance_output)))

        re_error_final = tf.reduce_sum(re_error_final)

        return re_error_final, inputul_mean_masked, inputul_var_masked, masked_outputul, list_bds_mean_masked, list_bds_masked_outputul    
    '''


