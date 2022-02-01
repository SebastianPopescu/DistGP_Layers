# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
from collections import defaultdict
import sys
import time
import argparse
import os
import math
from losses import *
from data_processing import *
from network_architectures import *
from conditional_GP import *
from propagate_layers import *
import nibabel as nib
import pandas as pd
from metrics import *
DTYPE=tf.float32
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#########################################
############ Helper functions ###########
#########################################

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

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def create_objects(num_data, num_test,
    dim_input, dim_output, num_iterations, num_layers, dim_layers, dim_convolutionally_warped_gp,
    num_inducing, type_var, 
    num_batch, num_samples_testing, 
    dim_filters, num_strides, 
    name_dataset, import_model, iteration_restored, learning_rate,
    dim_input_channels, num_patches_per_subject,
    num_subjects, num_averaged_gradients, use_bayesian_deep_supervision,
    testing_time, data_type, name_dataset_to_import, affine, multi_kernel, Z_init, use_masks):

    loss_functions_object = loss_functions(num_data = num_data,
        num_test = num_test,
        dim_input = dim_input,
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_layers = num_layers, 
        dim_layers = dim_layers, 
        dim_convolutionally_warped_gp = dim_convolutionally_warped_gp,
        num_inducing = num_inducing, 
        type_var = type_var, 
        num_batch = num_batch, 
        num_samples_testing = num_samples_testing, 
        dim_filters = dim_filters, 
        num_strides = num_strides, 
        name_dataset = name_dataset, 
        import_model = import_model, 
        iteration_restored = iteration_restored, 
        learning_rate = learning_rate,
        dim_input_channels= dim_input_channels, 
        num_patches_per_subject = num_patches_per_subject,
        num_subjects = num_subjects, 
        num_averaged_gradients = num_averaged_gradients, 
        use_bayesian_deep_supervision = use_bayesian_deep_supervision,
        testing_time = testing_time, 
        data_type = data_type, 
        name_dataset_to_import = name_dataset_to_import, 
        affine = affine, 
        multi_kernel = multi_kernel,
        Z_init = Z_init,
        use_masks = use_masks)

    propagate_layers_object = propagate_layers(num_data = num_data,
        num_test = num_test,
        dim_input = dim_input,
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_layers = num_layers, 
        dim_layers = dim_layers, 
        dim_convolutionally_warped_gp = dim_convolutionally_warped_gp,
        num_inducing = num_inducing, 
        type_var = type_var, 
        num_batch = num_batch, 
        num_samples_testing = num_samples_testing, 
        dim_filters = dim_filters, 
        num_strides = num_strides, 
        name_dataset = name_dataset, 
        import_model = import_model, 
        iteration_restored = iteration_restored, 
        learning_rate = learning_rate,
        dim_input_channels = dim_input_channels, 
        num_patches_per_subject = num_patches_per_subject,
        num_subjects = num_subjects, 
        num_averaged_gradients = num_averaged_gradients, 
        use_bayesian_deep_supervision = use_bayesian_deep_supervision,
        testing_time = testing_time, 
        data_type = data_type, 
        name_dataset_to_import = name_dataset_to_import, 
        affine = affine, 
        multi_kernel = multi_kernel,
        Z_init = Z_init,
        use_masks = use_masks)

    conditional_GP_object = conditional_GP(num_data = num_data,
        num_test = num_test,
        dim_input = dim_input,
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_layers = num_layers, 
        dim_layers = dim_layers, 
        dim_convolutionally_warped_gp = dim_convolutionally_warped_gp,
        num_inducing = num_inducing, 
        type_var = type_var, 
        num_batch = num_batch, 
        num_samples_testing = num_samples_testing, 
        dim_filters = dim_filters, 
        num_strides = num_strides, 
        name_dataset = name_dataset, 
        import_model = import_model, 
        iteration_restored = iteration_restored, 
        learning_rate = learning_rate,
        dim_input_channels = dim_input_channels, 
        num_patches_per_subject = num_patches_per_subject,
        num_subjects = num_subjects, 
        num_averaged_gradients = num_averaged_gradients, 
        use_bayesian_deep_supervision = use_bayesian_deep_supervision,
        testing_time = testing_time, 
        data_type = data_type, 
        name_dataset_to_import = name_dataset_to_import, 
        affine = affine, 
        multi_kernel = multi_kernel,
        Z_init = Z_init, 
        use_masks = use_masks)

    network_architectures_object=  network_architectures(num_data = num_data,
        num_test = num_test,
        dim_input = dim_input,
        dim_output = dim_output, 
        num_iterations = num_iterations, 
        num_layers = num_layers, 
        dim_layers = dim_layers, 
        dim_convolutionally_warped_gp = dim_convolutionally_warped_gp,
        num_inducing = num_inducing, 
        type_var = type_var, 
        num_batch = num_batch, 
        num_samples_testing = num_samples_testing, 
        dim_filters = dim_filters, 
        num_strides = num_strides, 
        name_dataset = name_dataset, 
        import_model = import_model, 
        iteration_restored = iteration_restored, 
        learning_rate = learning_rate,
        dim_input_channels = dim_input_channels, 
        num_patches_per_subject = num_patches_per_subject,
        num_subjects = num_subjects, 
        num_averaged_gradients = num_averaged_gradients, 
        use_bayesian_deep_supervision = use_bayesian_deep_supervision,
        testing_time = testing_time, 
        data_type = data_type, 
        name_dataset_to_import = name_dataset_to_import, 
        affine = affine, 
        multi_kernel = multi_kernel,
        Z_init = Z_init,
        use_masks = use_masks)

    return loss_functions_object, propagate_layers_object, conditional_GP_object, network_architectures_object


def train(session, 
    X_training, Y_training, mask_train, loss_weights, bds_masks, bds_outputul,
    X_training_feed, Y_training_feed, Y_training_mask_feed, loss_weights_np, bds_masks_np, bds_outputul_np, use_masks,
    num_averaged_gradients, cost, data_fit_cost, kl_cost, _train_op, list_data_fit_cost, _grad_op, _gradients, _grad_placeholders):

    feed_dict = {
        X_training: X_training_feed,
        Y_training: Y_training_feed,
        loss_weights : loss_weights_np}
    if use_masks:
        feed_dict.update({mask_train : Y_training_mask_feed})
    
    for _ in range(len(bds_outputul_np)):
        if use_masks:
            feed_dict.update({
                bds_masks[_] : bds_masks_np[_]})

        feed_dict.update({
            bds_outputul[_] : bds_outputul_np[_]
        })

    if num_averaged_gradients == 1:

        loss_np, data_fit_cost_np, kl_cost_np, _, list_data_fit_cost_np = session.run([cost, data_fit_cost, kl_cost, _train_op, list_data_fit_cost], 
            feed_dict=feed_dict)

    else:
        
        loss_np, data_fit_cost_np, kl_cost_np, grads, list_data_fit_cost_np = session.run([cost, data_fit_cost, kl_cost, _grad_op, list_data_fit_cost],
            feed_dict=feed_dict)
        _gradients.append(grads)
        
        if len(_gradients) == num_averaged_gradients:
            for i, placeholder in enumerate(_grad_placeholders):
                    feed_dict[placeholder] = np.stack([g[i] for g in _gradients], axis=0).mean(axis=0)
            session.run(_train_op, feed_dict=feed_dict)

    return loss_np, data_fit_cost_np, kl_cost_np, list_data_fit_cost_np



def Dist_GP_activated_NN(num_data, num_test,
    dim_input, dim_output, num_iterations, num_layers, dim_layers, dim_convolutionally_warped_gp,
    num_inducing, type_var, 
    num_batch, num_samples_testing, 
    dim_filters, num_strides, 
    name_dataset, import_model, iteration_restored, learning_rate,
    dim_input_channels, num_patches_per_subject,
    num_subjects, num_averaged_gradients, use_bayesian_deep_supervision,
    testing_time, data_type, name_dataset_to_import, affine, multi_kernel,
    X_training, Y_training, X_testing, Y_testing, masks_training, masks_testing,
    Z_init, use_masks):

    
    tf.disable_v2_behavior()

    gpu_options = tf.GPUOptions(allow_growth = True)
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
        log_device_placement = False, gpu_options = gpu_options))

    dim_last_layer = dim_input
    ######## get size of last layer ####
    if use_masks:
        bds_masks = []
    bds_outputul = []
    list_dim_last_layer = [] #### to be used for BDS routine 

    for _ in range(num_layers-1):

        print(dim_last_layer)
        if _==1:
            dim_last_layer = (dim_last_layer - 2*dim_filters[_]) // num_strides[_] +2     
        else:
            dim_last_layer = (dim_last_layer - dim_filters[_]) // num_strides[_] +1 

        list_dim_last_layer.append(dim_last_layer)

        if data_type=='2D':
            
            ### this is for Bayesian Deep Supervision routine ###
            if use_masks:           
                bds_masks.append(tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, 1), name = 'mask_training_bds_layer_'+str(_+1)))
            bds_outputul.append(tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_output), name = 'Y_training_bds_layer_'+str(_+1)))
        
        elif data_type=='3D':
            
            ### this is for Bayesian Deep Supervision routine ###
            if use_masks:
                bds_masks.append(tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, 
                    1), name = 'mask_training_bds_layer_'+str(_+1)))
            bds_outputul.append(tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, 
                dim_output), name = 'Y_training_bds_layer_'+str(_+1)))


    print('*************************************************************************************')
    print('dimension of last layer')
    print(dim_last_layer)

    if data_type == '2D':

        ####### Training placeholders ####
        X_training_ph = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input_channels), name = 'X_training')
        Y_training_ph = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_output), name = 'Y_training')

        ####### Testing placeholders ####
        X_testing_ph = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input_channels), name = 'X_testing')
        Y_testing_ph = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_output), name = 'Y_testing')

        if use_masks:
            #### Mask placeholders ####
            mask_train_ph = tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, 1), name='mask_training')
            mask_test_ph = tf.placeholder(tf.bool, shape=(None, dim_last_layer,  dim_last_layer, 1), name='mask_testing')

    elif data_type == '3D':

        ####### Training placeholders ####
        X_training_ph = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input, dim_input_channels), name = 'X_training')
        Y_training_ph = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, dim_output), name = 'Y_training')

        ####### Testing placeholders ####
        X_testing_ph = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input, dim_input_channels), name = 'X_testing')
        Y_testing_ph = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, dim_output), name = 'Y_testing')

        if use_masks:
            #### Mask placeholders ####
            mask_train_ph = tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, 1), name='mask_training')
            mask_test_ph = tf.placeholder(tf.bool, shape=(None, dim_last_layer,  dim_last_layer, dim_last_layer, 1), name='mask_testing')

    loss_weights = tf.placeholder(DTYPE, shape=(num_layers, ), name='dirichlet_concentration_parameters')

    ### create objects ###
        
    loss_functions_object, propagate_layers_object, conditional_GP_object, network_architectures_object = create_objects(num_data, num_test,
        dim_input, dim_output, num_iterations, num_layers, dim_layers, dim_convolutionally_warped_gp,
        num_inducing, type_var, 
        num_batch, num_samples_testing, 
        dim_filters, num_strides, 
        name_dataset, import_model, iteration_restored, learning_rate,
        dim_input_channels, num_patches_per_subject,
        num_subjects, num_averaged_gradients, use_bayesian_deep_supervision,
        testing_time, data_type, name_dataset_to_import, affine, multi_kernel, Z_init, use_masks)

    if not testing_time:


        ### using DeepMedic baseline model ###
        list_pred_mean_training, list_pred_var_training, kl_cost = network_architectures_object.baseline_deep_medic(inputul = X_training_ph,
            propagate_layers_object = propagate_layers_object, conditional_GP_object = conditional_GP_object, training_time = True, full_cov=False)

        if use_masks:
            data_fit_cost, f_sampled_training, y_training_masked, f_sampled_training_bds, y_training_masked_bds, list_data_fit_cost = loss_functions_object.classification(outputul = Y_training_ph, 
                list_inputul_mean = list_pred_mean_training,
                list_inputul_var = list_pred_var_training,
                masks = mask_train_ph, loss_weights = loss_weights, bds_masks = bds_masks, bds_outputul = bds_outputul, use_masks = use_masks)
        else:
            data_fit_cost, f_sampled_training, y_training, f_sampled_training_bds, y_training_bds, list_data_fit_cost = loss_functions_object.classification(outputul = Y_training_ph, 
                list_inputul_mean = list_pred_mean_training,
                list_inputul_var = list_pred_var_training,
                masks = None, loss_weights = loss_weights, bds_masks = None, bds_outputul = bds_outputul, use_masks = use_masks)

        cost = data_fit_cost - kl_cost

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = learning_rate
        
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.1, staircase=True)
        # Passing global_step to minimize() will increment it at each step.

        if num_averaged_gradients == 1:

            with tf.control_dependencies(extra_update_ops):
                _train_op = tf.train.AdamOptimizer(learning_rate).minimize(-cost, global_step = global_step)

        else:

            # here 'train_op' only applies gradients passed via placeholders stored
            # in 'grads_placeholders. The gradient computation is done with 'grad_op'.
            optimizer = tf.train.AdamOptimizer(learning_rate)
            with tf.control_dependencies(extra_update_ops):

                all_variables = tf.trainable_variables()
                grads_and_vars = optimizer.compute_gradients(-cost, var_list = all_variables)
                for sth in grads_and_vars:
                    print(sth)
                grads_and_vars = [(tf.clip_by_value(grad, -100.0, 100.0), var) for grad, var in grads_and_vars]
                
            avg_grads_and_vars = []
            _grad_placeholders = []
            for grad, var in grads_and_vars:
                grad_ph = tf.placeholder(grad.dtype, grad.shape)
                _grad_placeholders.append(grad_ph)
                avg_grads_and_vars.append((grad_ph, var))

            _grad_op = [x[0] for x in grads_and_vars]
            _train_op = optimizer.apply_gradients(avg_grads_and_vars)
            _gradients = [] # list to store gradients

        #### Metrics for Training Set ####

        ###### make sure probabilites are between 0 and 1 ####
        pred_mean_train_bds = []

        if dim_output==1:
            pred_mean_train = inv_probit(f_sampled_training)
            for plm in range(num_layers-1):
                pred_mean_train_bds.append(inv_probit(f_sampled_training_bds[plm]))
            
        else:

            pred_mean_train = tf.nn.softmax(f_sampled_training)

            for plm in range(num_layers-1):
                pred_mean_train_bds.append(tf.nn.softmax(f_sampled_training_bds[plm]))


        ##########################
        #### Accuracy metrics ####
        ##########################

        correct_pred_training_DKL_bds =defaultdict()

        if dim_output==1:
            if use_masks:
                correct_pred_training_DKL = tf.equal(tf.round(pred_mean_train), tf.round(y_training_masked))
            else:
                correct_pred_training_DKL = tf.equal(tf.round(pred_mean_train), tf.round(y_training))              
    
            for plm in range(num_layers-1):
                if use_masks:
                    correct_pred_training_DKL_bds[plm] = tf.equal(tf.round(pred_mean_train_bds[plm]), tf.round(y_training_masked_bds[plm]))
                else:
                    correct_pred_training_DKL_bds[plm] = tf.equal(tf.round(pred_mean_train_bds[plm]), tf.round(y_training_bds[plm]))                  
        else:
            if use_masks:
                correct_pred_training_DKL = tf.equal(tf.argmax(pred_mean_train, 1), tf.argmax(y_training_masked, 1))
            else:
                correct_pred_training_DKL = tf.equal(tf.argmax(pred_mean_train, 1), tf.argmax(y_training, 1))               
            for plm in range(num_layers-1):
                if use_masks:
                    correct_pred_training_DKL_bds[plm] = tf.equal(tf.argmax(pred_mean_train_bds[plm],-1), tf.argmax(y_training_masked_bds[plm], -1))
                else:
                    correct_pred_training_DKL_bds[plm] = tf.equal(tf.argmax(pred_mean_train_bds[plm],-1), tf.argmax(y_training_bds[plm], -1))

        accuracy_training_DKL = tf.reduce_mean(tf.cast(correct_pred_training_DKL,DTYPE))
        accuracy_training_DKL_bds = defaultdict()

        for plm in range(num_layers-1):
            accuracy_training_DKL_bds[plm] = tf.reduce_mean(tf.cast(correct_pred_training_DKL_bds[plm],DTYPE)) 


        if dim_output==1:

            #### Dice score metrics for Binary Classification ####
            if use_masks:
                kkt = y_training_masked
            else:
                kkt = y_training
            dice_score_train_DKL = dice_score(predicted_labels = pred_mean_train,
                labels = kkt, dim_output = dim_output,
                type_unet = data_type)
        
            dice_score_train_DKL_bds =defaultdict()
            for plm in range(num_layers-1):

                if use_masks:
                    kkt =  y_training_masked_bds[plm]
                else:
                    kkt =  y_training_bds[plm]

                dice_score_train_DKL_bds[plm] = dice_score(predicted_labels = pred_mean_train_bds[plm],
                    labels = kkt, dim_output = dim_output,
                    type_unet = data_type)

        else:

            if use_masks:
                kkt = y_training_masked
            else:
                kkt = y_training
            #### Dice score metrics for Multi-Class Classification ####
            dice_score_train_DKL = dice_score_multiclass(predicted_labels = pred_mean_train,
                labels = kkt, num_classes = dim_output,
                type_unet = data_type)

    
            dice_score_train_DKL_bds =defaultdict()
            for plm in range(num_layers-1):

                if use_masks:
                    kkt =  y_training_masked_bds[plm]
                else:
                    kkt =  y_training_bds[plm]

                dice_score_train_DKL_bds[plm] = dice_score_multiclass(predicted_labels = pred_mean_train_bds[plm],
                    labels = kkt, num_classes = dim_output,
                    type_unet = data_type)

        #######################################
        #### save them to tensorboard file ####
        #######################################

        tf.summary.scalar('acc_train_DKL', tf.squeeze(accuracy_training_DKL))

        for plm in range(num_layers-1): 
            tf.summary.scalar('acc_train_bds_layer_'+str(plm), tf.squeeze(accuracy_training_DKL_bds[plm]))

        classes_to_iterate_over = dim_output
        if classes_to_iterate_over==1:
            classes_to_iterate_over+=1

        for plm in range(num_layers-1):

            for num_class in range(classes_to_iterate_over):
                tf.summary.scalar('dice_train_bds_layer_'+str(plm)+'_num_class_'+str(num_class),
                    tf.squeeze(dice_score_train_DKL_bds[plm][num_class]))

        for num_class in range(classes_to_iterate_over):
            tf.summary.scalar('dice_train_DKL_num_class_'+str(num_class),
                tf.squeeze(dice_score_train_DKL[num_class]))


        tf.summary.scalar('log_lik_DKL', tf.squeeze(data_fit_cost))
        tf.summary.scalar('kl_dkl', tf.squeeze(kl_cost))


    if multi_kernel:
        text_to_add='_multi_kernel_version_'
    else:
        text_to_add=''

    #### get testing set network #### 

    list_f_mean_testing, list_f_var_epistemic_testing, list_f_var_distributional_testing = network_architectures_object.baseline_deep_medic_uncertainty_decomposed(inputul = X_testing_ph, 
        propagate_layers_object=propagate_layers_object,
        conditional_GP_object=conditional_GP_object, full_cov=False )
    list_f_var_testing = [plm1 + plm2 for plm1,plm2 in zip(list_f_var_epistemic_testing, list_f_var_distributional_testing)]

    if testing_time:

        if dim_output==1:
            f_mean_testing = inv_probit(list_f_mean_testing[-1])     
        else:
            f_mean_testing = tf.nn.softmax(list_f_mean_testing[-1])

    else:
        
        ##### Metrics for Tensorboard ##############

        ###### make sure probabilites are between 0 and 1 ####

        if dim_output==1:
            f_mean_testing = inv_probit(list_f_mean_testing[-1])
            log_lik_testing = bernoulli(list_f_mean_testing[-1], Y_testing_ph)

        else:


            log_lik_testing = multiclass_helper(tf.reshape(list_f_mean_testing[-1],[-1,dim_output]), tf.reshape(Y_testing_ph, [-1,dim_output]))
            f_mean_testing = tf.nn.softmax(list_f_mean_testing[-1])

        #### Accuracy metrics ####

        if dim_output==1:
            correct_pred_testing_DKL = tf.equal(tf.round(f_mean_testing), tf.round(Y_testing_ph))
        else:
            correct_pred_testing_DKL = tf.equal(tf.argmax(f_mean_testing, 1), tf.argmax(Y_testing_ph, 1))

        accuracy_testing_DKL = tf.reduce_mean(tf.cast(correct_pred_testing_DKL, DTYPE))
        
        if dim_output==1:

            #### Dice score metrics for Binary Classification ####
            
            dice_score_test_DKL = dice_score(predicted_labels = f_mean_testing,
                labels = Y_testing_ph, dim_output = dim_output,
                type_unet = data_type)

        else:

            #### Dice score metrics for Multi-Class Classification ####

            dice_score_test_DKL = dice_score_multiclass(predicted_labels = f_mean_testing,
                labels = Y_testing_ph, num_classes = dim_output,
                type_unet = data_type)
    
        #### save them to tensorboard file ####


        tf.summary.scalar('acc_test_DKL', tf.squeeze(accuracy_testing_DKL))

        classes_to_iterate_over = dim_output
        if classes_to_iterate_over==1:
            classes_to_iterate_over+=1

        for num_class in range(classes_to_iterate_over):
            tf.summary.scalar('dice_test_DKL_num_class_'+str(num_class),
                tf.squeeze(dice_score_test_DKL[num_class]))

        ######### Sanity check tensorboard histogram logs ########
        ######### Get Mean and Variance from each hidden layer ###
        for _ in range(num_layers-1):

            variable_summaries(var = list_f_mean_testing[_], name = 'mean_layer_'+str(_+1))
            variable_summaries(var = list_f_var_testing[_], name = 'variance_layer_'+str(_+1))



    where_to_save = str(name_dataset)+str(text_to_add)+'/num_layers_'+str(num_layers)+'/dim_layers_'+str(dim_layers[1])+'/num_inducing_'+str(num_inducing[1])

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./tensorboard/'+where_to_save)

    saver = tf.train.Saver()


    #### grab just the variables from previous models ####

    where_to_grab = str(name_dataset_to_import)+str(text_to_add)+'/num_layers_'+str(num_layers)+'/dim_layers_'+str(dim_layers[1])+'/num_inducing_'+str(num_inducing[1])+'/iteration_'+str(iteration_restored)

    if import_model==True:

        print('attempting to grab the model at')
        print('./saved_models/'+where_to_grab)    
        saver_grabber = tf.train.Saver()
        saver_grabber.restore(sess, tf.train.latest_checkpoint('./saved_models/'+where_to_grab))  
        print('Grabed checkpoint')

    else:
    
        sess.run(tf.global_variables_initializer())

    tvars = tf.trainable_variables()

    for var in tvars:
        print(var.name)  # Prints the name of the variable alongside its value.

    graph = tf.get_default_graph()

    if testing_time:

        #################################################################
        #### get the final segmentations over the entire testing set ####
        #################################################################

        cmd = 'mkdir -p ./image_segmentations/'+where_to_save
        os.system(cmd)

        if data_type=='2D':
            num_testing_subjects = X_testing.shape[0]
        elif data_type=='3D':
            num_testing_subjects = len(X_testing.keys())

        for _ in range(num_testing_subjects):

            print('*******************************')
            print('we are at image num '+str(_))
            print('********************************')

            if data_type=='2D':

                current_image = X_testing[_,...]
                if use_masks:
                    current_mask = masks_testing[_,...]
                shape_of_data = X_testing[_,...].shape
                current_label = Y_testing[_,...]
                
            elif data_type=='3D':

                current_image = X_testing[_]
                if use_masks:
                    current_mask = masks_testing[_]
                shape_of_data = X_testing[_].shape
                current_label = Y_testing[_]

            size_cube_input1 = dim_input//2
            size_cube_output1 = dim_last_layer//2
            size_cube_input2 = dim_input - size_cube_input1
            size_cube_output2 = dim_last_layer - size_cube_output1
            print(size_cube_input1)
            print(size_cube_output1)
            print(size_cube_input2)
            print(size_cube_output2)

            if data_type=='2D':
            
                current_label = np.expand_dims(current_label, axis=-1)
                if use_masks:
                    patches, patches_labels, shape_of_data_after_padding, current_mask_padded, current_image_padded, current_label_padded = extract_2D_cubes_input_seg(input_image = current_image, output_image = current_label,
                        semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                        semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                        dim_output = dim_output, mask = current_mask, use_masks = use_masks)
                else:
                    patches, patches_labels, shape_of_data_after_padding, current_image_padded, current_label_padded = extract_2D_cubes_input_seg(input_image = current_image, output_image = current_label,
                        semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                        semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                        dim_output = dim_output, mask = None, use_masks = use_masks)

            elif data_type=='3D':

                if use_masks:
                    patches, patches_labels, shape_of_data_after_padding, current_mask_padded, current_label_padded, current_image_padded = extract_3D_cubes_input_seg(input_image = current_image, output_image = current_label,
                        semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                        semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                        dim_output = dim_output, mask = current_mask, use_masks = use_masks)
                else:
                    patches, patches_labels, shape_of_data_after_padding,  current_label_padded, current_image_padded = extract_3D_cubes_input_seg(input_image = current_image, output_image = current_label,
                        semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                        semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                        dim_output = dim_output, mask = None, use_masks = use_masks)

            print('size of what we got from custom made non-overlapping 3D cuube extraction')
            print(patches.shape)
            print(patches_labels.shape)
        
            num_batch_test_time = 5
            num_iterate_over = patches.shape[0] 
            num_batches = num_iterate_over // num_batch_test_time
            lista_batches = [np.arange(kkt*num_batch_test_time,(kkt+1)*num_batch_test_time) for kkt in range(num_batches-1)]
            lista_batches.append(np.arange((num_batches-1)*num_batch_test_time, num_iterate_over))
            predictions_testing_np = []

            t1 = time.time()
            mean_segmentation = []
            var_epistemic_segmentation = []
            var_distributional_segmentation = []

            for i_batch in range(num_batches):

                pred_mean_now, pred_var_epistemic_now, pred_var_distributional_now = sess.run([f_mean_testing, list_f_var_epistemic_testing[-1], 
                    list_f_var_distributional_testing[-1]], feed_dict={X_testing_ph : patches[lista_batches[i_batch]],
                    Y_testing_ph : patches_labels[lista_batches[i_batch]]})
                
                print('*** size of segmentation moments ***')
                print(pred_mean_now.shape)
                print(pred_var_epistemic_now.shape)
                print(pred_var_distributional_now.shape)
                if data_type=='2D':
                    mean_segmentation.append(pred_mean_now.reshape((-1, dim_last_layer, dim_last_layer, dim_output)))
                    var_epistemic_segmentation.append(pred_var_epistemic_now.reshape((-1, dim_last_layer, dim_last_layer, dim_output)))
                    var_distributional_segmentation.append(pred_var_distributional_now.reshape((-1, dim_last_layer, dim_last_layer, dim_output)))

                elif data_type=='3D':

                    mean_segmentation.append(pred_mean_now.reshape((-1, dim_last_layer, dim_last_layer, dim_last_layer, dim_output)))
                    var_epistemic_segmentation.append(pred_var_epistemic_now.reshape((-1, dim_last_layer, dim_last_layer, dim_last_layer, dim_output)))
                    var_distributional_segmentation.append(pred_var_distributional_now.reshape((-1, dim_last_layer, dim_last_layer, dim_last_layer, dim_output)))

            mean_segmentation = np.concatenate(mean_segmentation, axis = 0)
            var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = 0)
            var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = 0)

            if data_type=='2D':

                mean_segmentation = [np.expand_dims(unblockshaped(arr = mean_segmentation[..., mata], h = shape_of_data_after_padding[0],
                    w = shape_of_data_after_padding[1]),axis=-1) for mata in range(dim_output)]
                mean_segmentation = np.concatenate(mean_segmentation, axis = -1)

                var_epistemic_segmentation = [np.expand_dims(unblockshaped(arr = var_epistemic_segmentation[..., mata], 
                    h = shape_of_data_after_padding[0], w = shape_of_data_after_padding[1]),
                    axis=-1) for mata in range(dim_output)]
                var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = -1)

                var_distributional_segmentation = [np.expand_dims(unblockshaped(arr = var_distributional_segmentation[..., mata], 
                    h = shape_of_data_after_padding[0], w = shape_of_data_after_padding[1]),
                    axis=-1) for mata in range(dim_output)]
                var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = -1)

                ### compute predictive entropy ###
                predictive_entropy = -np.multiply(mean_segmentation, np.log(mean_segmentation+1e-6))
                predictive_entropy = np.sum(predictive_entropy, axis=-1, keepdims=False)


                print(mean_segmentation.shape)
                print(var_epistemic_segmentation.shape)
                print(var_distributional_segmentation.shape)
                print(predictive_entropy.shape)

                ###################################
                ######## Manual annotation ########
                ###################################
    
                cmd='mkdir -p ./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/manual_annotation'
                os.system(cmd)

                plt.imshow(current_image_padded[...,0])
                plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/manual_annotation/simple.png')
                plt.close()	
        
                for kkt_pe_bat in range(dim_output):
                        
                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    if use_masks:
                        plt.imshow(apply_mask(var_distributional_segmentation[...,kkt_pe_bat], 
                            current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    else:
                        plt.imshow(var_distributional_segmentation[...,kkt_pe_bat], cmap='jet', alpha=0.6)                       
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/manual_annotation/distributional_uncertainty_'+str(kkt_pe_bat)+'.png')
                    plt.close()	
            
                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    if use_masks:
                        plt.imshow(apply_mask(var_epistemic_segmentation[...,kkt_pe_bat], 
                            current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    else:
                        plt.imshow(var_epistemic_segmentation[...,kkt_pe_bat], cmap='jet', alpha=0.6)                       
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/manual_annotation/epistemic_uncertainty_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    if use_masks:
                        plt.imshow(apply_mask(mean_segmentation[...,kkt_pe_bat], 
                            current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    else:
                        plt.imshow(mean_segmentation[...,kkt_pe_bat], cmap='jet', alpha=0.6)                        
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/manual_annotation/mean_segmentation_'+str(kkt_pe_bat)+'.png')
                    plt.close()	
        
                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    if use_masks:
                        plt.imshow(apply_mask(predictive_entropy, current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    else:
                        plt.imshow(predictive_entropy, cmap='jet', alpha=0.6)   
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/manual_annotation/predictive_entropy_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

                '''
                ### TODO -- update also these lines to acocunt for use_masks ###

                ################################
                ####### Raw Image ##############
                ################################
                cmd='mkdir -p ./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image'
                os.system(cmd)

                print('size of current label padded')
                print(current_label_padded.shape)

                plt.imshow(current_label_padded)

                plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image/simple.png')
                plt.close()	
        
                for kkt_pe_bat in range(dim_output):
            
                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(var_distributional_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image/distributional_uncertainty_'+str(kkt_pe_bat)+'.png')
                    plt.close()	
            
                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(var_epistemic_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image/epistemic_uncertainty_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(mean_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image/mean_segmentation_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(predictive_entropy, current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image/predictive_entropy_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

                    fig,axes = plt.subplots(nrows=1, ncols=5)

                    fig.set_figheight(100)
                    fig.set_figwidth(500)
                    axes[0].imshow(current_label_padded)


                    axes[1].imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    im1 = axes[1].imshow(apply_mask(mean_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    #axes[1].colorbar()
                    #fig.colorbar(im1, ax = axes[1], orientation='vertical')

                    axes[2].imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    im2 = axes[2].imshow(apply_mask(predictive_entropy, current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    #axes[2].colorbar()
                    #fig.colorbar(im2, ax = axes[2], orientation='vertical')

                    axes[3].imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    im3 = axes[3].imshow(apply_mask(var_distributional_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    #axes[3].colorbar()
                    #fig.colorbar(im3, ax = axes[3], orientation='vertical')    

                    axes[4].imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    im4 = axes[4].imshow(apply_mask(var_epistemic_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet', alpha=0.6)
                    #axes[4].colorbar()
                    #fig.colorbar(im4, ax = axes[4], orientation='vertical')
                    plt.tight_layout()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/raw_image/cum_trebuie.png')
                    plt.close()	

                for kkt_pe_bat in range(dim_output):

                    plt.imshow(apply_mask(var_distributional_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet')
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/distributional_uncertainty_'+str(kkt_pe_bat)+'.png')
                    plt.close()	
            
                    plt.imshow(apply_mask(var_epistemic_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet')
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/epistemic_uncertainty_'+str(kkt_pe_bat)+'.png')
                    plt.close()	


                    plt.imshow(apply_mask(mean_segmentation[...,kkt_pe_bat], 
                        current_mask_padded[...,0]), cmap='jet')
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/mean_segmentation_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

                    plt.imshow(apply_mask(predictive_entropy, current_mask_padded[...,0]), cmap='jet')
                    plt.colorbar()
                    plt.savefig('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/predictive_entropy_'+str(kkt_pe_bat)+'.png')
                    plt.close()	

              
                    #### save as csf file ###

                    print(predictive_entropy.shape)
                    print(mean_segmentation.shape)
                    print(var_epistemic_segmentation.shape)
                    print(var_distributional_segmentation.shape)
                    print(current_label_padded.shape)

                    df_dict = { 'Predictive Entropy' : predictive_entropy.ravel(),
                        'Mean' : mean_segmentation.ravel(),
                        'Epistemic' : var_epistemic_segmentation.ravel(),
                        'Distributional' : var_distributional_segmentation.ravel(),
                        'Input_0' : current_label_padded[...,0].ravel(),
                        'Input_1' : current_label_padded[...,1].ravel(),
                        'Input_2' : current_label_padded[...,2].ravel()
                    }

                    df = pd.DataFrame(df_dict)
                    df.to_csv('./image_segmentations/'+where_to_save+'/image_num_'+str(_)+'/results.csv', index=False)
                '''
            elif data_type=='3D':

                mean_segmentation = [np.expand_dims(uncubify(mean_segmentation[...,kkt], 
                    (shape_of_data_after_padding[0], shape_of_data_after_padding[1], 
                    shape_of_data_after_padding[2])),axis=-1) for kkt in range(dim_output)]
                mean_segmentation = np.concatenate(mean_segmentation, axis = -1)

                var_epistemic_segmentation = [np.expand_dims(uncubify(var_epistemic_segmentation[...,kkt], 
                    (shape_of_data_after_padding[0], shape_of_data_after_padding[1], 
                    shape_of_data_after_padding[2])),axis=-1) for kkt in range(dim_output) ]
                var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = -1)
                
                var_distributional_segmentation = [np.expand_dims(uncubify(var_distributional_segmentation[...,kkt], 
                    (shape_of_data_after_padding[0], shape_of_data_after_padding[1], 
                    shape_of_data_after_padding[2])),axis=-1) for kkt in range(dim_output)]
                var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = -1)
                
                if use_masks:
                    current_mask_padded = np.reshape(current_mask_padded, 
                        (shape_of_data_after_padding[0], shape_of_data_after_padding[1], shape_of_data_after_padding[2]))

                ### TODO eventually -- need to mask the above in the final nifti files ###
        
                cmd='mkdir -p ./image_segmentations/'+where_to_save+'/image_num_'+str(_)
                os.system(cmd)
                prefix = './image_segmentations/'+where_to_save+'/image_num_'+str(_)

                predictive_entropy = -np.multiply(mean_segmentation, np.log(mean_segmentation+1e-6))
                predictive_entropy = np.sum(predictive_entropy, axis=-1, keepdims=False)

                if dim_output>1:

                    ### get distributional differental entropy ###
                    differential_entropy_distributional =  0.5 * np.sum(np.log(var_distributional_segmentation+1e-27), axis=-1, keepdims=True)        

                    img = nib.Nifti1Image(differential_entropy_distributional, affine)
                    nib.save(img,prefix+'/differential_entropy_distributional.nii.gz' )
    
                    ### get epistemic differential entropy ###
                    differential_entropy_epistemic =  0.5 * np.sum(np.log(var_epistemic_segmentation+1e-27), axis=-1, keepdims=True)

                    img = nib.Nifti1Image(differential_entropy_epistemic, affine)
                    nib.save(img,prefix+'/differential_entropy_epistemic.nii.gz' )
        
                img = nib.Nifti1Image(mean_segmentation, affine)
                nib.save(img,prefix+'/mean_segmentation.nii.gz' )
        
                img = nib.Nifti1Image(var_epistemic_segmentation, affine)
                nib.save(img,prefix+'/epistemic_var_segmentation.nii.gz' )
        
                img = nib.Nifti1Image(var_distributional_segmentation, affine)
                nib.save(img,prefix+'/distributional_var_segmentation.nii.gz' )

                img = nib.Nifti1Image(current_image_padded, affine)
                nib.save(img,prefix+'/input.nii.gz' )
        
                img = nib.Nifti1Image(current_label_padded, affine)
                nib.save(img,prefix+'/labels.nii.gz' )
        
                img = nib.Nifti1Image(predictive_entropy, affine)
                nib.save(img,prefix+'/predictive_entropy.nii.gz' )
        
    else:

        for i in np.arange(50001,50001+num_iterations):

            if i < 2500:

                ##### all reconstruction parameters get the same weight  #####

                loss_weights_np = [0.5/num_layers for _ in range(num_layers-1)]
                loss_weights_np.append(0.5)
                #print(loss_weights_np)

            elif i > 2500 * (num_layers-1):

                #### just the main re loss gets a weight #####

                loss_weights_np = [0.0 for _ in range(num_layers-1)]
                loss_weights_np.append(1.0)
                #print(loss_weights_np)

            else:

                ### the first bayesian deep supervision re loss get zero weight, whereas the remainder get the weight divided equally ####
                first_bayesian_layer = int(i / 2500)
                loss_weights_np = [0.0 for _ in range(first_bayesian_layer)]
                loss_weights_np.extend([0.5/(num_layers-first_bayesian_layer-1) for _ in range(num_layers-first_bayesian_layer-1)])
                loss_weights_np.append(0.5)

            costul_actual_overall = []
            re_cost_actual_overall = [] 
            kl_cost_actual_overall = []
            re_cost_actual_list_overall = defaultdict()
            for kkt in range(num_layers):

                re_cost_actual_list_overall[kkt] = []

            _gradients  = []
            for separate_minibatch in range(num_averaged_gradients):

                if data_type=='2D':
                    if use_masks:
                        input_batches, output_batches, mask_batches, output_batches_bds, mask_batches_bds = extract_2d_blocks_training(inputul = X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = masks_training, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)
                    else:
                        input_batches, output_batches, output_batches_bds = extract_2d_blocks_training(inputul = X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = None, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)

                elif data_type=='3D':
                    if use_masks: 
                        input_batches, output_batches, mask_batches, output_batches_bds, mask_batches_bds = extract_3d_blocks_training(inputul = X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = masks_training, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)
                    else:
                        input_batches, output_batches, output_batches_bds = extract_3d_blocks_training(inputul = X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = None, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)                       

                if use_masks:

                    costul_actual, re_cost_actual, kl_cost_actual, list_costul_actual = train(session = sess, X_training_feed = input_batches, 
                        Y_training_feed = output_batches, Y_training_mask_feed = mask_batches, 
                        loss_weights_np = loss_weights_np,
                        bds_masks_np = mask_batches_bds,
                        bds_outputul_np = output_batches_bds,
                        num_averaged_gradients =  num_averaged_gradients,
                        cost = cost,
                        data_fit_cost = data_fit_cost,
                        kl_cost = kl_cost,
                        _train_op = _train_op,
                        list_data_fit_cost = list_data_fit_cost,
                        _grad_op = _grad_op,
                        _gradients = _gradients,
                        _grad_placeholders = _grad_placeholders,
                        X_training = X_training_ph,
                        Y_training = Y_training_ph,
                        mask_train = mask_train_ph,
                        loss_weights = loss_weights ,
                        bds_masks = bds_masks,
                        bds_outputul = bds_outputul,
                        use_masks = use_masks)
                else:

                    costul_actual, re_cost_actual, kl_cost_actual, list_costul_actual = train(session = sess, X_training_feed = input_batches, 
                        Y_training_feed = output_batches, Y_training_mask_feed = None, 
                        loss_weights_np = loss_weights_np,
                        bds_masks_np = None,
                        bds_outputul_np = output_batches_bds,
                        num_averaged_gradients =  num_averaged_gradients,
                        cost = cost,
                        data_fit_cost = data_fit_cost,
                        kl_cost = kl_cost,
                        _train_op = _train_op,
                        list_data_fit_cost = list_data_fit_cost,
                        _grad_op = _grad_op,
                        _gradients = _gradients,
                        _grad_placeholders = _grad_placeholders,
                        X_training = X_training_ph,
                        Y_training = Y_training_ph,
                        mask_train = None,
                        loss_weights = loss_weights ,
                        bds_masks = None,
                        bds_outputul = bds_outputul,
                        use_masks = use_masks)

                costul_actual_overall.append(costul_actual)
                re_cost_actual_overall.append(re_cost_actual)
                kl_cost_actual_overall.append(kl_cost_actual)
                for kkt in range(num_layers):
                    re_cost_actual_list_overall[kkt].append(list_costul_actual[kkt])

            costul_actual = np.mean(costul_actual_overall)
            re_cost_actual = np.mean(re_cost_actual_overall)
            kl_cost_actual = np.mean(kl_cost_actual_overall)
            for kkt in range(num_layers):
                re_cost_actual_list_overall[kkt] = np.mean(re_cost_actual_list_overall[kkt])
            
            #train_writer.add_summary(summary,i)			
            
            if i % 500 == 0 and i != 0:

                if data_type=='2D':

                    #########################################
                    #### get mini_batch for training data ###
                    #########################################

                    if use_masks:

                        input_batches_training, output_batches_training, mask_batches_training, bds_output_batches_training, bds_mask_batches_training = extract_2d_blocks_training(inputul =X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = masks_training, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)
                
                        ########################################
                        #### get mini_batch for testing data ###
                        ########################################

                        input_batches_testing, output_batches_testing, mask_batches_testing, bds_output_batches_testing, bds_mask_batches_testing = extract_2d_blocks_training(inputul = X_testing, 
                            outputul = Y_testing, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = masks_testing, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)
              
                    else:

                        input_batches_training, output_batches_training, bds_output_batches_training = extract_2d_blocks_training(inputul =X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = None, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)                      

                        input_batches_testing, output_batches_testing,  bds_output_batches_testing = extract_2d_blocks_training(inputul = X_testing, 
                            outputul = Y_testing, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = None, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)

                
                elif data_type=='3D':

                    if use_masks:

                        #########################################
                        #### get mini_batch for training data ###
                        #########################################

                        input_batches_training, output_batches_training, mask_batches_training, bds_output_batches_training, bds_mask_batches_training = extract_3d_blocks_training(inputul =X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = masks_training, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)

                        ########################################
                        #### get mini_batch for testing data ###
                        ########################################

                        input_batches_testing, output_batches_testing, mask_batches_testing, bds_output_batches_testing, bds_mask_batches_testing = extract_3d_blocks_training(inputul = X_testing, 
                            outputul = Y_testing, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = masks_testing, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)
                    else:

                        #########################################
                        #### get mini_batch for training data ###
                        #########################################

                        input_batches_training, output_batches_training, bds_output_batches_training= extract_3d_blocks_training(inputul =X_training, 
                            outputul = Y_training, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = None, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)

                        ########################################
                        #### get mini_batch for testing data ###
                        ########################################

                        input_batches_testing, output_batches_testing,  bds_output_batches_testing= extract_3d_blocks_training(inputul = X_testing, 
                            outputul = Y_testing, iteration = i, block_size_input = dim_input, block_size_output = dim_last_layer,
                            dim_output = dim_output, num_subjects = num_subjects,
                            num_patches_per_subject = num_patches_per_subject, masks = None, list_dim_last_layer = list_dim_last_layer, use_masks = use_masks)


                feed_dict={X_training_ph : input_batches_training,
                    Y_training_ph : output_batches_training,
                    X_testing_ph : input_batches_testing,
                    Y_testing_ph : output_batches_testing,
                    loss_weights : loss_weights_np
                }

                if use_masks:
                    feed_dict.update({mask_train_ph : mask_batches_training, mask_test_ph : mask_batches_testing})

                for _ in range(len(bds_output_batches_training)):
                    if use_masks:
                        feed_dict.update({
                            bds_masks[_] : bds_mask_batches_training[_]
                        })

                    feed_dict.update({
                        bds_outputul[_] : bds_output_batches_training[_]
                    })

                summary = sess.run(merged, 
                    feed_dict
                    )
                train_writer.add_summary(summary, i)			

                '''
                #### Sanity check for testing data ####

                summary, batches_mean_prediction, batches_var_prediction, log_lik_testing_np = sess.run([merged, list_f_mean_testing[-1], list_f_var_testing[-1], log_lik_testing],
                    feed_dict
                    )
                train_writer.add_summary(summary, i)			

                if data_type=='2D':

                    ################################
                    ################################
                    ### Sanity Check for 2D data ###
                    ################################

                    cmd = 'mkdir -p ./sanity_check'
                    os.system(cmd)

                    cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)
                    os.system(cmd)

                    ### save txt ###
                    np.savetxt('./sanity_check/iteration_'+str(i)+'/log_lik_testing.txt', log_lik_testing_np.ravel())

                    for plm in range(input_batches_testing.shape[0]):

                        _= plm
                        cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)
                        os.system(cmd)
            
                        muie = mask_batches_testing[_,...,0].astype(float)

                        plt.imshow(muie, cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.png')
                        plt.close()	
            
                        text_de_scris = str(np.max((mask_batches_testing[_,...,0])))
                        with open('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/max_mask.txt','w') as f:
                            f.write(text_de_scris)
            
                        text_de_scris = str(np.min((mask_batches_testing[_,...,0])))
                        with open('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/min_mask.txt','w') as f:
                            f.write(text_de_scris)

                        plt.imshow(input_batches_testing[_,...,0], cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/input.png')
                        plt.close()	

                        for kkt_pe_bat in range(dim_output):
                                
                            plt.imshow(output_batches_testing[_,...,kkt_pe_bat], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_pe_bat)+'.png')
                            plt.close()	

                            plt.imshow(batches_mean_prediction[_,...,kkt_pe_bat], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_pe_bat)+'.png')
                            plt.close()	

                            plt.imshow(batches_var_prediction[_,...,kkt_pe_bat], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_pe_bat)+'.png')
                            plt.close()	

                elif data_type=='3D':

                    ################################
                    ################################
                    ### Sanity Check for 3D data ###
                    ################################

                    cmd = 'mkdir -p ./sanity_check'
                    os.system(cmd)

                    cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)
                    os.system(cmd)

                    ### save txt ###
                    np.savetxt('./sanity_check/iteration_'+str(i)+'/log_lik_testing.txt', log_lik_testing_np.ravel())


                    for plm in range(input_batches_testing.shape[0]):

                        _= plm
                        cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)
                        os.system(cmd)

                        #### we can take a slice on the second dimension ####
                        arr = np.arange(dim_last_layer)
                        np.random.shuffle(arr)
                        slice_num = arr[0]

                        plt.imshow(input_batches_testing[_,:,slice_num,:,0], cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/input.png')
                        plt.close()	

                        ### save txt ###
                        np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/input.txt',input_batches_testing[_,:,slice_num,:,0].ravel())

                        muie = mask_batches_testing[_,:,slice_num,:,0].astype(float)
                        plt.imshow(muie, cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.png')
                        plt.close()	

                        ### save txt ###
                        np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.txt',mask_batches_testing[_,:,slice_num,:,0].ravel())

                        ### get the cross entropy here  and save it in a text file ###
                            
                        plt.imshow(np.sum(output_batches_testing[_,:,slice_num,:,:],axis=-1), cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_overall.png')
                        plt.close()	

                        for kkt_class in range(dim_output):
                                
                            plt.imshow(output_batches_testing[_,:,slice_num,:,kkt_class], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_class)+'.png')
                            plt.close()	

                            ### save txt ###
                            np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_class)+'.txt',output_batches_testing[_,:,slice_num,:,kkt_class].ravel())

                            plt.imshow(batches_mean_prediction[_,:,slice_num,:,kkt_class], cmap='seismic')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_class)+'.png')
                            plt.close()	

                            ### save txt ###
                            np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_class)+'.txt',batches_mean_prediction[_,:,slice_num,:,kkt_class].ravel())
                            
                            plt.imshow(batches_var_prediction[_,:,slice_num,:,kkt_class], cmap='Spectral')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_class)+'.png')
                            plt.close()	

                            ### save txt ###
                            np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_class)+'.txt',batches_var_prediction[_,:,slice_num,:,kkt_class].ravel())
    
                #### Sanity check for training data ####

                batches_mean_prediction, batches_var_prediction, log_lik_testing_np = sess.run([list_pred_mean_training[-1], list_pred_var_training[-1]],
                    feed_dict
                    )

                if data_type=='2D':

                    ################################
                    ################################
                    ### Sanity Check for 2D data ###
                    ################################

                    cmd = 'mkdir -p ./sanity_check_training'
                    os.system(cmd)

                    cmd = 'mkdir -p ./sanity_check_training/iteration_'+str(i)
                    os.system(cmd)

                    ### save txt ###
                    np.savetxt('./sanity_check_training/iteration_'+str(i)+'/log_lik_testing.txt', log_lik_testing_np.ravel())

                    for plm in range(input_batches_training.shape[0]):

                        _= plm
                        cmd = 'mkdir -p ./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)
                        os.system(cmd)
            
                        muie = mask_batches_training[_,...,0].astype(float)

                        plt.imshow(muie, cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.png')
                        plt.close()	
            
                        text_de_scris = str(np.max((mask_batches_training[_,...,0])))
                        with open('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/max_mask.txt','w') as f:
                            f.write(text_de_scris)
            
                        text_de_scris = str(np.min((mask_batches_training[_,...,0])))
                        with open('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/min_mask.txt','w') as f:
                            f.write(text_de_scris)

                        plt.imshow(input_batches_training[_,...,0], cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/input.png')
                        plt.close()	

                        for kkt_pe_bat in range(dim_output):
                                
                            plt.imshow(output_batches_training[_,...,kkt_pe_bat], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_pe_bat)+'.png')
                            plt.close()	

                            plt.imshow(batches_mean_prediction[_,...,kkt_pe_bat], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_pe_bat)+'.png')
                            plt.close()	

                            plt.imshow(batches_var_prediction[_,...,kkt_pe_bat], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check_training/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_pe_bat)+'.png')
                            plt.close()	

                elif data_type=='3D':

                    ################################
                    ################################
                    ### Sanity Check for 3D data ###
                    ################################

                    cmd = 'mkdir -p ./sanity_check'
                    os.system(cmd)

                    cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)
                    os.system(cmd)

                    ### save txt ###
                    np.savetxt('./sanity_check/iteration_'+str(i)+'/log_lik_testing.txt', log_lik_testing_np.ravel())


                    for plm in range(input_batches_testing.shape[0]):

                        _= plm
                        cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)
                        os.system(cmd)

                        #### we can take a slice on the second dimension ####
                        arr = np.arange(dim_last_layer)
                        np.random.shuffle(arr)
                        slice_num = arr[0]

                        plt.imshow(input_batches_testing[_,:,slice_num,:,0], cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/input.png')
                        plt.close()	

                        ### save txt ###
                        np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/input.txt',input_batches_testing[_,:,slice_num,:,0].ravel())

                        muie = mask_batches_testing[_,:,slice_num,:,0].astype(float)
                        plt.imshow(muie, cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.png')
                        plt.close()	

                        ### save txt ###
                        np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.txt',mask_batches_testing[_,:,slice_num,:,0].ravel())

                        ### get the cross entropy here  and save it in a text file ###
                            
                        plt.imshow(np.sum(output_batches_testing[_,:,slice_num,:,:],axis=-1), cmap='gray')
                        plt.colorbar()
                        plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_overall.png')
                        plt.close()	

                        for kkt_class in range(dim_output):
                                
                            plt.imshow(output_batches_testing[_,:,slice_num,:,kkt_class], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_class)+'.png')
                            plt.close()	

                            ### save txt ###
                            np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_class)+'.txt',output_batches_testing[_,:,slice_num,:,kkt_class].ravel())

                            plt.imshow(batches_mean_prediction[_,:,slice_num,:,kkt_class], cmap='seismic')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_class)+'.png')
                            plt.close()	

                            ### save txt ###
                            np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_class)+'.txt',batches_mean_prediction[_,:,slice_num,:,kkt_class].ravel())
                            
                            plt.imshow(batches_var_prediction[_,:,slice_num,:,kkt_class], cmap='Spectral')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_class)+'.png')
                            plt.close()	

                            ### save txt ###
                            np.savetxt('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_class)+'.txt',batches_var_prediction[_,:,slice_num,:,kkt_class].ravel())
    
                '''                

                ###############################################################################################################################
                ###############################################################################################################################
                
            if i % 10000 == 0 and i!=0:

            
                cmd = 'mkdir -p ./saved_models/'+str(name_dataset)+str(text_to_add)+'/num_layers_'+str(num_layers)+'/dim_layers_'+str(dim_layers[1])+'/num_inducing_'+str(num_inducing[1])+'/iteration_'+str(i)
                os.system(cmd)  

                saver.save(sess, './saved_models/'+str(name_dataset)+str(text_to_add)+'/num_layers_'+str(num_layers)+'/dim_layers_'+str(dim_layers[1])+'/num_inducing_'+str(num_inducing[1])+'/iteration_'+str(i)+'/saved_DeepConvGP', global_step=i)  
                print('Saved checkpoint')


            print('******* Start ******')
            print('at iteration '+str(i) + ' we have nll : '+str(costul_actual) + 're cost :'+str(re_cost_actual)+' kl cost :'+str(kl_cost_actual))
            for kkt in range(num_layers):

                print(re_cost_actual_list_overall[kkt])
            print('******* End ********')

