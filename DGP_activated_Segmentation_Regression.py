# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import defaultdict
import sys
import time
import argparse
import os
import math
from losses import *
from build_predict import *
from data_processing import *
from network_architectures import *
import nibabel as nib
DTYPE=tf.float32

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

def one_hot_encoder(input):

    object = np.zeros(shape=(input.shape[0],10))
    for i in range(input.shape[0]):
        object[i,int(input[i])] = 1.0
    return object	

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

class Dist_GP_activated_NN(object):

    def __init__(self, num_data, num_test,
        dim_input, dim_output, num_iterations, num_layers, dim_layers, dim_convolutionally_warped_gp,
        num_inducing, type_var, 
        num_batch, num_samples_testing, 
        dim_filters, num_strides, 
        name_dataset, import_model, iteration_restored, learning_rate,
        dim_input_channels, num_patches_per_subject,
        num_subjects, num_averaged_gradients, use_bayesian_deep_supervision,
        testing_time, data_type, name_dataset_to_import, affine, mean_function):

        self.mean_function = mean_function
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
        self.num_layers=num_layers
        self.dim_layers=dim_layers
        self.dim_input_channels = dim_input_channels

        dim_last_layer = dim_input
        ######## get size of last layer ####
        self.bds_masks = []
        self.bds_outputul = []
        self.list_dim_last_layer = [] #### to be used for BDS routine 

        tf.disable_v2_behavior()

        for _ in range(num_layers-1):

            print(dim_last_layer)
            dim_last_layer = (dim_last_layer - dim_filters[_]) // num_strides[_] +1 
            self.list_dim_last_layer.append(dim_last_layer)

            if self.data_type=='2D':
                
                ### this is for Bayesian Deep Supervision routine ###
                self.bds_masks.append(tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, 1), name = 'mask_training_bds_layer_'+str(_+1)))
                self.bds_outputul.append(tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_output), name = 'Y_training_bds_layer_'+str(_+1)))
            
            elif self.data_type=='3D':
               
                ### this is for Bayesian Deep Supervision routine ###
                self.bds_masks.append(tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, 
                    1), name = 'mask_training_bds_layer_'+str(_+1)))
                self.bds_outputul.append(tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, 
                    dim_output), name = 'Y_training_bds_layer_'+str(_+1)))
            
        self.dim_last_layer = dim_last_layer

        print('*************************************************************************************')
        print('dimension of last layer')
        print(dim_last_layer)
        #print('number of patches last layer ')

        if self.data_type == '2D':

            ####### Training placeholders ####
            self.X_training = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input_channels), name = 'X_training')
            self.Y_training = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_output), name = 'Y_training')

            ####### Testing placeholders ####
            self.X_testing = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input_channels), name = 'X_testing')
            self.Y_testing = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_output), name = 'Y_testing')

            #### Mask placeholders ####
            self.mask_train = tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, 1), name='mask_training')
            self.mask_test = tf.placeholder(tf.bool, shape=(None, dim_last_layer,  dim_last_layer, 1), name='mask_testing')

        elif self.data_type == '3D':

            ####### Training placeholders ####
            self.X_training = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input, dim_input_channels), name = 'X_training')
            self.Y_training = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, dim_output), name = 'Y_training')

            ####### Testing placeholders ####
            self.X_testing = tf.placeholder(DTYPE, shape=(None, dim_input, dim_input, dim_input, dim_input_channels), name = 'X_testing')
            self.Y_testing = tf.placeholder(DTYPE, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, dim_output), name = 'Y_testing')

            #### Mask placeholders ####
            self.mask_train = tf.placeholder(tf.bool, shape=(None, dim_last_layer, dim_last_layer, dim_last_layer, 1), name='mask_training')
            self.mask_test = tf.placeholder(tf.bool, shape=(None, dim_last_layer,  dim_last_layer, dim_last_layer, 1), name='mask_testing')

        #######################################
        #### for bayesian deep supervision ####
        #######################################

        self.loss_weights = tf.placeholder(tf.float32, shape=(self.num_layers,), name='dirichlet_concentration_parameters')

        #############################
        #############################
        #############################

    def setup_train(self):

        ### using DeepMedic baseline model ###
        pred_mean_training, pred_var_training, list_pred_mean_training, list_pred_var_training = baseline_deep_medic(inputul = self.X_training,
            num_layers = self.num_layers, dim_layers = self.dim_layers, dim_filters = self.dim_filters, 
            num_strides = self.num_strides, data_type = self.data_type, full_cov=False)

        self.data_fit_cost, self.f_mean_training, self.f_var_training, self.y_training_masked, self.f_mean_training_bds, self.y_training_masked_bds = regression_error_CNN(inputul = [pred_mean_training, pred_var_training], 
            outputul = self.Y_training, list_inputul_mean = list_pred_mean_training, list_inputul_var = list_pred_var_training,
            dim_output = self.dim_output, num_layers = self.num_layers, num_data = self.num_data, dim_layers = self.dim_layers,
            num_strides = self.num_strides, dim_filters = self.dim_filters, masks = self.mask_train, use_bayesian_deep_supervision = self.use_bayesian_deep_supervision,
            loss_weights = self.loss_weights, bds_masks = self.bds_masks, bds_outputul = self.bds_outputul, data_type = self.data_type)

        self.kl_cost = KL_error(num_layers = self.num_layers, dim_layers = self.dim_layers)
        self.cost =  self.data_fit_cost - self.kl_cost

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 10000, 0.1, staircase=True)
        # Passing global_step to minimize() will increment it at each step.

        if self.num_averaged_gradients == 1:

            with tf.control_dependencies(extra_update_ops):
                self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.cost, global_step = self.global_step)

        else:

            # here 'train_op' only applies gradients passed via placeholders stored
            # in 'grads_placeholders. The gradient computation is done with 'grad_op'.
            optimizer = tf.train.AdamOptimizer(learning_rate)
            with tf.control_dependencies(extra_update_ops):


                all_variables = tf.trainable_variables()
                grads_and_vars = optimizer.compute_gradients(-self.cost, var_list = all_variables)
                for sth in grads_and_vars:
                    print(sth)
                grads_and_vars = [(tf.clip_by_value(grad, -100.0, 100.0), var) for grad, var in grads_and_vars]
                
            avg_grads_and_vars = []
            self._grad_placeholders = []
            for grad, var in grads_and_vars:
                grad_ph = tf.placeholder(grad.dtype, grad.shape)
                self._grad_placeholders.append(grad_ph)
                avg_grads_and_vars.append((grad_ph, var))

            self._grad_op = [x[0] for x in grads_and_vars]
            self._train_op = optimizer.apply_gradients(avg_grads_and_vars)
            self._gradients = [] # list to store gradients

    def train(self, session, X_training_feed, Y_training_feed, Y_training_mask_feed, loss_weights_np, bds_masks_np, bds_outputul_np):

        feed_dict = {
            self.X_training: X_training_feed,
            self.Y_training: Y_training_feed,
            self.mask_train : Y_training_mask_feed,
            self.loss_weights : loss_weights_np
            }
        for _ in range(len(bds_masks_np)):
            feed_dict.update({
                self.bds_masks[_] : bds_masks_np[_],
                self.bds_outputul[_] : bds_outputul_np[_]
            })
        if self.num_averaged_gradients == 1:

            loss, data_fit_cost, kl_cost, _ = session.run([self.cost, self.data_fit_cost, self.kl_cost, self._train_op], 
                feed_dict=feed_dict)

        else:
            
            loss, data_fit_cost, kl_cost, grads = session.run([self.cost, self.data_fit_cost, self.kl_cost, self._grad_op],
                feed_dict=feed_dict)
            self._gradients.append(grads)
            
            if len(self._gradients) == self.num_averaged_gradients:
                for i, placeholder in enumerate(self._grad_placeholders):
                      feed_dict[placeholder] = np.stack([g[i] for g in self._gradients], axis=0).mean(axis=0)
                session.run(self._train_op, feed_dict=feed_dict)
                self._gradients = []

        return loss, data_fit_cost, kl_cost



    def session_TF(self, X_training, Y_training, X_testing, Y_testing, masks_training, masks_testing):

        #### above variables are dictionaries ####

        #####################
        #### get session ####
        #####################

        gpu_options = tf.GPUOptions(allow_growth = True)
        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
            log_device_placement = False, gpu_options = gpu_options))

        ######################
        #### get losses ######
        ######################

        self.setup_train()

        #####################################################################################################
        #### create Representation Learning framework based on baseline DeepMedic model for testing data ####
        #####################################################################################################

        f_mean_testing_rl, f_var_testing_rl, list_f_mean_testing, list_f_var_testing = baseline_deep_medic(inputul = self.X_testing, 
            num_layers = self.num_layers, dim_layers = self.dim_layers, dim_filters = self.dim_filters, 
            num_strides = self.num_strides, data_type = self.data_type, full_cov=False)

        f_sampled_testing, f_mean_testing, f_var_testing, f_var_epistemic_testing, f_var_distributional_testing = build_predict_uncertainity_decomposition_CNN(inputul = [f_mean_testing_rl, f_var_testing_rl],
            num_layers = self.num_layers, dim_layers = self.dim_layers, dim_filters = self.dim_filters, 
            num_strides = self.num_strides, data_type = self.data_type)

        f_sampled_testing+=self.mean_function
        f_mean_testing+=self.mean_function

        ############################################
        ##### Metrics for Tensorboard ##############
        ############################################

        #####################################
        #### Mean Absolute Error metrics ####
        #####################################

        mae_training_DKL_bds =defaultdict()

        mae_training_DKL = tf.reduce_mean(tf.abs(self.f_mean_training - self.y_training_masked))
        mae_testing_DKL = tf.reduce_mean(tf.abs(f_mean_testing -  self.Y_testing))

        for plm in range(self.num_layers-1):

            mae_training_DKL_bds[plm] = tf.reduce_mean(tf.abs(self.f_mean_training_bds[plm] - self.y_training_masked_bds[plm]))

 
        mae_training_DKL = tf.reduce_mean(tf.cast(mae_training_DKL,DTYPE))
        mae_testing_DKL = tf.reduce_mean(tf.cast(mae_testing_DKL, DTYPE))
    

        #######################################
        #### save them to tensorboard file ####
        #######################################

        tf.summary.scalar('mae_test_DKL', tf.squeeze(mae_testing_DKL))
        tf.summary.scalar('mae_train_DKL', tf.squeeze(mae_training_DKL))
        for plm in range(self.num_layers-1):
            
            tf.summary.scalar('mae_train_bds_layer_'+str(plm), tf.squeeze(mae_training_DKL_bds[plm]))

        ######### Sanity check tensorboard histogram logs ########
        ######### Get Mean and Variance from each hidden layer ###
        for _ in range(self.num_layers-1):

            variable_summaries(var = list_f_mean_testing[_], name = 'mean_layer_'+str(_+1))
            variable_summaries(var = list_f_var_testing[_], name = 'variance_layer_'+str(_+1))

        tf.summary.scalar('log_lik_DKL', tf.squeeze(self.data_fit_cost))
        tf.summary.scalar('kl_dkl', tf.squeeze(self.kl_cost))

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./tensorboard_'+str(self.name_dataset)+'/num_layers_'+str(self.num_layers)+'/dim_layers_'+str(self.dim_layers[1])+'/num_inducing_'+str(self.num_inducing[1]))

        saver = tf.train.Saver()

        #### grab just the variables from previous models ####

        if self.import_model==True:

            print('attempting to grab the model at')
            print('./saved_models_'+str(self.name_dataset_to_import)+'/num_layers_'+str(self.num_layers)+'/dim_layers_'+str(self.dim_layers[1])+'/num_inducing_'+str(self.num_inducing[1])+'/iteration_'+str(self.iteration_restored))    
            saver_grabber = tf.train.Saver()
            saver_grabber.restore(sess, tf.train.latest_checkpoint('./saved_models_'+str(self.name_dataset_to_import)+'/num_layers_'+str(self.num_layers)+'/dim_layers_'+str(self.dim_layers[1])+'/num_inducing_'+str(self.num_inducing[1])+'/iteration_'+str(self.iteration_restored)))  
            print('Grabed checkpoint')

        else:
        
            sess.run(tf.global_variables_initializer())

        tvars = tf.trainable_variables()

        for var in tvars:
            print(var.name)  # Prints the name of the variable alongside its value.

        graph = tf.get_default_graph()

        if self.testing_time:

            #### TODO -- this needs to be updated with the nex patch extraction functions for Regression problems ###
            #################################################################
            #### get the final segmentations over the entire testing set ####
            #################################################################

            cmd = 'mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)
            os.system(cmd)

            for _ in range(X_testing.shape[0]):

                print('*******************************')
                print('we are at image num '+str(_))
                print('********************************')

                current_image = X_testing[_,...]
                current_mask = masks_testing[_,...]

                #### We pad each brain scans so that we can take non-overlapping cubic blocks over it #####
                shape_of_data = X_testing[_,...].shape

                #current_mask = np.logical_not(np.equal(current_image,np.zeros_like(current_image)))
                current_label = Y_testing[_,...]
                
                size_cube_input1 = self.dim_input//2
                size_cube_output1 = self.dim_last_layer//2
                size_cube_input2 = self.dim_input - size_cube_input1
                size_cube_output2 = self.dim_last_layer - size_cube_output1
                print(size_cube_input1)
                print(size_cube_output1)
                print(size_cube_input2)
                print(size_cube_output2)

                if self.data_type=='2D':
                        
                    patches, patches_labels, shape_of_data_after_padding, current_mask_padded, current_image_padded, current_label_padded = extract_2D_cubes_input_seg(input_image = current_image, output_image = current_label,
                        semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                        semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                        dim_output = self.dim_output, mask = current_mask)

                elif self.data_type=='3D':

                    patches, patches_labels, shape_of_data_after_padding, current_mask_padded, current_image_padded, current_label_padded = extract_3D_cubes_input_seg(input_image = current_image, output_image = current_label,
                        semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                        semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                        dim_output = self.dim_output, mask = current_mask)

                print('size of what we got from custom made non-overlapping 3D cuube extraction')
                print(patches.shape)
                print(patches_labels.shape)
            
                num_batch_test_time = 10
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

                    pred_mean_now, pred_var_epistemic_now, pred_var_distributional_now = sess.run([f_mean_testing, f_var_epistemic_testing, 
                        f_var_distributional_testing], feed_dict={self.X_testing:patches[lista_batches[i_batch]],
                        self.Y_testing:patches_labels[lista_batches[i_batch]]})
                    print('*** size of segmentation moments ***')
                    print(pred_mean_now.shape)
                    print(pred_var_epistemic_now.shape)
                    print(pred_var_distributional_now.shape)
                    if self.data_type=='2D':

                        mean_segmentation.append(pred_mean_now.reshape((-1, self.dim_last_layer, self.dim_last_layer)))
                        var_epistemic_segmentation.append(pred_var_epistemic_now.reshape((-1, self.dim_last_layer, self.dim_last_layer)))
                        var_distributional_segmentation.append(pred_var_distributional_now.reshape((-1, self.dim_last_layer, self.dim_last_layer)))

                    elif self.data_type=='3D':

                        mean_segmentation.append(pred_mean_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_last_layer)))
                        var_epistemic_segmentation.append(pred_var_epistemic_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_last_layer)))
                        var_distributional_segmentation.append(pred_var_distributional_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_last_layer)))


                mean_segmentation = np.concatenate(mean_segmentation, axis = 0)
                var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = 0)
                var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = 0)

                t2 = time.time()
                print('how much time it takes per subject')
                timer(t1,t2)

                if self.data_type=='2D':

                    mean_segmentation = unblockshaped(mean_segmentation, 
                        shape_of_data_after_padding[0], shape_of_data_after_padding[1])
                    var_epistemic_segmentation = unblockshaped(var_epistemic_segmentation, 
                        shape_of_data_after_padding[0], shape_of_data_after_padding[1])
                    var_distributional_segmentation = unblockshaped(var_distributional_segmentation, 
                        shape_of_data_after_padding[0], shape_of_data_after_padding[1])
                    current_mask_padded = np.reshape(current_mask_padded, (shape_of_data_after_padding[0], shape_of_data_after_padding[1]))

                elif self.data_type=='3D':

                    mean_segmentation = uncubify(mean_segmentation, 
                        (shape_of_data_after_padding[0], shape_of_data_after_padding[1], shape_of_data_after_padding[2]))
                    var_epistemic_segmentation = uncubify(var_epistemic_segmentation, 
                        (shape_of_data_after_padding[0], shape_of_data_after_padding[1], shape_of_data_after_padding[2]))
                    var_distributional_segmentation = uncubify(var_distributional_segmentation, 
                        (shape_of_data_after_padding[0], shape_of_data_after_padding[1], shape_of_data_after_padding[2]))
                    current_mask_padded = np.reshape(current_mask_padded, 
                        (shape_of_data_after_padding[0], shape_of_data_after_padding[1], shape_of_data_after_padding[2]))

                print('*** shape of data after re-arranging it ****')
                print(mean_segmentation.shape)
                print(var_epistemic_segmentation.shape)
                print(var_distributional_segmentation.shape)
                
                cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)
                os.system(cmd)					

                if self.data_type=='2D':

                    ###################################
                    ######## Manual annotation ########
                    ###################################

                    cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/manual_annotation'
                    os.system(cmd)

                    plt.imshow(current_label_padded)
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/manual_annotation/simple.png')
                    plt.close()	
            
                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(var_distributional_segmentation, current_mask_padded), cmap='magma', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/manual_annotation/distributional_uncertainty.png')
                    plt.close()	
            
                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(var_epistemic_segmentation, current_mask_padded), cmap='magma', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/manual_annotation/epistemic_uncertainty.png')
                    plt.close()	

                    plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                    plt.imshow(apply_mask(mean_segmentation, current_mask_padded), cmap='magma', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/manual_annotation/mean_segmentation.png')
                    plt.close()	

    

                    ####### Raw Image ##############

                    cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/raw_image'
                    os.system(cmd)

                    plt.imshow(current_image_padded[...,0])
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/raw_image/simple.png')
                    plt.close()	
            
                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    plt.imshow(apply_mask(var_distributional_segmentation, current_mask_padded), cmap='magma', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/raw_image/distributional_uncertainty.png')
                    plt.close()	
            
                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    plt.imshow(apply_mask(var_epistemic_segmentation, current_mask_padded), cmap='magma', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/raw_image/epistemic_uncertainty.png')
                    plt.close()	

                    plt.imshow(current_image_padded[...,0], alpha=0.6)
                    plt.imshow(apply_mask(mean_segmentation, current_mask_padded), cmap='magma', alpha=0.6)
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/raw_image/mean_segmentation.png')
                    plt.close()	


                    ################################
                    ####### Mean Segmentations #####
                    ################################

                    cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/sampled_seg'
                    os.system(cmd)

                    plt.imshow(apply_mask(mean_segmentation, current_mask_padded), cmap='gray')
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/sampled_seg/mean_segmentation.png')
                    plt.colorbar()
                    plt.close()	

                    ############################################
                    ####### Distributional Uncertainty #########
                    ############################################

                    cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/uncertainty'
                    os.system(cmd)
            
                    plt.imshow(apply_mask(var_distributional_segmentation, current_mask_padded), cmap='magma')
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/uncertainty/distributional_uncertainty.png')
                    plt.close()	
                    
                    plt.imshow(apply_mask(var_epistemic_segmentation, current_mask_padded), cmap='magma')
                    plt.colorbar()
                    plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/uncertainty/epistemic_uncertainty.png')
                    plt.close()	                

                elif self.data_type=='3D':
         

                    img = nib.Nifti1Image(mean_segmentation, self.affine)
                    nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/mean_segmentation.nii.gz' )
            
                    img = nib.Nifti1Image(var_epistemic_segmentation, self.affine)
                    nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/epistemic_var_segmentation.nii.gz' )
            
                    img = nib.Nifti1Image(var_distributional_segmentation, self.affine)
                    nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/image_num_'+str(_)+'/distributional_var_segmentation.nii.gz' )

           
        else:

            for i in range(self.num_iterations):


                if i < 50000:

                    ##### all reconstruction parameters get the same weight  #####

                    loss_weights_np = [0.5/self.num_layers for _ in range(self.num_layers-1)]
                    loss_weights_np.append(0.5)
                    #print(loss_weights_np)

                elif i > 50000 * (self.num_layers-1):

                    #### just the main re loss gets a weight #####

                    loss_weights_np = [0.0 for _ in range(self.num_layers-1)]
                    loss_weights_np.append(1.0)
                    #print(loss_weights_np)

                else:

                    ### the first bayesian deep supervision re loss get zero weight, whereas the remainder get the weight divided equally ####
                    first_bayesian_layer = int(i / 50000)
                    loss_weights_np = [0.0 for _ in range(first_bayesian_layer)]
                    loss_weights_np.extend([0.5/(self.num_layers-first_bayesian_layer-1) for _ in range(self.num_layers-first_bayesian_layer-1)])
                    loss_weights_np.append(0.5)

                costul_actual_overall = []
                re_cost_actual_overall = [] 
                kl_cost_actual_overall = []

                for separate_minibatch in range(self.num_averaged_gradients):

                    if self.data_type=='2D':
                            
                        input_batches, output_batches, mask_batches, output_batches_bds, mask_batches_bds = extract_2d_blocks_training(inputul = X_training, 
                            outputul = Y_training, iteration = i, block_size_input = self.dim_input, block_size_output = self.dim_last_layer,
                            dim_output = self.dim_output, num_subjects = self.num_subjects,
                            num_patches_per_subject = self.num_patches_per_subject, masks = masks_training, list_dim_last_layer = self.list_dim_last_layer)

                    elif self.data_type=='3D':
                          
                        input_batches, output_batches, mask_batches, output_batches_bds, mask_batches_bds = extract_3d_blocks_training_regression(inputul = X_training, 
                            outputul = Y_training, iteration = i, block_size_input = self.dim_input, block_size_output = self.dim_last_layer,
                            dim_output = self.dim_output, num_subjects = self.num_subjects,
                            num_patches_per_subject = self.num_patches_per_subject, masks = masks_training, list_dim_last_layer = self.list_dim_last_layer)

                    costul_actual, re_cost_actual, kl_cost_actual = self.train(session = sess, X_training_feed = input_batches, 
                        Y_training_feed = output_batches, Y_training_mask_feed = mask_batches, loss_weights_np = loss_weights_np,
                        bds_masks_np = mask_batches_bds,
                        bds_outputul_np = output_batches_bds)

                    costul_actual_overall.append(costul_actual)
                    re_cost_actual_overall.append(re_cost_actual)
                    kl_cost_actual_overall.append(kl_cost_actual)

                costul_actual = np.mean(costul_actual_overall)
                re_cost_actual = np.mean(re_cost_actual_overall)
                kl_cost_actual = np.mean(kl_cost_actual_overall)

                #train_writer.add_summary(summary,i)			
                
                if i % 100 == 0 and i != 0:

                    if self.data_type=='2D':

                        #########################################
                        #### get mini_batch for training data ###
                        #########################################

                        input_batches_training, output_batches_training, mask_batches_training, bds_output_batches_training, bds_mask_batches_training = extract_2d_blocks_training(inputul =X_training, 
                            outputul = Y_training, iteration = i, block_size_input = self.dim_input, block_size_output = self.dim_last_layer,
                            dim_output = self.dim_output, num_subjects = self.num_subjects,
                            num_patches_per_subject = self.num_patches_per_subject, masks = masks_training, list_dim_last_layer = self.list_dim_last_layer)

                        ########################################
                        #### get mini_batch for testing data ###
                        ########################################

                        input_batches_testing, output_batches_testing, mask_batches_testing, bds_output_batches_testing, bds_mask_batches_testing = extract_2d_blocks_training(inputul = X_testing, 
                            outputul = Y_testing, iteration = i, block_size_input = self.dim_input, block_size_output = self.dim_last_layer,
                            dim_output = self.dim_output, num_subjects = self.num_subjects,
                            num_patches_per_subject = self.num_patches_per_subject, masks = masks_testing, list_dim_last_layer = self.list_dim_last_layer)
                    
                    elif self.data_type=='3D':

                        #########################################
                        #### get mini_batch for training data ###
                        #########################################

                        input_batches_training, output_batches_training, mask_batches_training, bds_output_batches_training, bds_mask_batches_training = extract_3d_blocks_training_regression(inputul =X_training, 
                            outputul = Y_training, iteration = i, block_size_input = self.dim_input, block_size_output = self.dim_last_layer,
                            dim_output = self.dim_output, num_subjects = self.num_subjects,
                            num_patches_per_subject = self.num_patches_per_subject, masks = masks_training, list_dim_last_layer = self.list_dim_last_layer)

                        ########################################
                        #### get mini_batch for testing data ###
                        ########################################

                        input_batches_testing, output_batches_testing, mask_batches_testing, bds_output_batches_testing, bds_mask_batches_testing = extract_3d_blocks_training_regression(inputul = X_testing, 
                            outputul = Y_testing, iteration = i, block_size_input = self.dim_input, block_size_output = self.dim_last_layer,
                            dim_output = self.dim_output, num_subjects = self.num_subjects,
                            num_patches_per_subject = self.num_patches_per_subject, masks = masks_testing, list_dim_last_layer = self.list_dim_last_layer)
                    
                    feed_dict={self.X_training : input_batches_training,
                        self.Y_training : output_batches_training,
                        self.X_testing : input_batches_testing,
                        self.Y_testing : output_batches_testing,
                        self.mask_train : mask_batches_training,
                        self.mask_test : mask_batches_testing, 
                        self.loss_weights : loss_weights_np
                    }

                    for _ in range(len(bds_mask_batches_training)):
                        feed_dict.update({
                            self.bds_masks[_] : bds_mask_batches_training[_],
                            self.bds_outputul[_] : bds_output_batches_training[_]
                        })

                    summary, batches_mean_prediction, batches_var_prediction = sess.run([merged, f_mean_testing, f_var_testing],
                        feed_dict
                        )
                    train_writer.add_summary(summary, i)			
                    '''              
                    if self.data_type=='2D':

                        ################################
                        ################################
                        ### Sanity Check for 2D data ###
                        ################################

                        cmd = 'mkdir -p ./sanity_check'
                        os.system(cmd)

                        cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)
                        os.system(cmd)

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

                            for kkt_pe_bat in range(self.dim_output):
                                    
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

                    elif self.data_type=='3D':

                        ################################
                        ################################
                        ### Sanity Check for 3D data ###
                        ################################

                        cmd = 'mkdir -p ./sanity_check'
                        os.system(cmd)

                        cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)
                        os.system(cmd)

                        for plm in range(input_batches_testing.shape[0]):

                            _= plm
                            cmd = 'mkdir -p ./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)
                            os.system(cmd)

                            #### we can take a slice on the second dimension ####
                            arr = np.arange(self.dim_last_layer)
                            np.random.shuffle(arr)
                            slice_num = arr[0]


                            plt.imshow(input_batches_testing[_,:,slice_num,:,0], cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/input.png')
                            plt.close()	

                            muie = mask_batches_testing[_,:,slice_num,:,0].astype(float)
                            plt.imshow(muie, cmap='gray')
                            plt.colorbar()
                            plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mask.png')
                            plt.close()	
    
                            for kkt_class in range(self.dim_output):
                                    
                                plt.imshow(output_batches_testing[_,:,slice_num,:,kkt_class], cmap='gray')
                                plt.colorbar()
                                plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/output_'+str(kkt_class)+'.png')
                                plt.close()	
                
                                plt.imshow(batches_mean_prediction[_,:,slice_num,:,kkt_class], cmap='gray')
                                plt.colorbar()
                                plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/mean_prediction_'+str(kkt_class)+'.png')
                                plt.close()	

                                plt.imshow(batches_var_prediction[_,:,slice_num,:,kkt_class], cmap='gray')
                                plt.colorbar()
                                plt.savefig('./sanity_check/iteration_'+str(i)+'/batch_'+str(plm)+'/var_prediction_'+str(kkt_class)+'.png')
                                plt.close()	
                    '''                       

                    ###############################################################################################################################
                    ###############################################################################################################################
                    
                if i % 50000 == 0 and i!=0:

                    cmd = 'mkdir -p ./saved_models_'+str(self.name_dataset)+'/num_layers_'+str(self.num_layers)+'/dim_layers_'+str(self.dim_layers[1])+'/num_inducing_'+str(self.num_inducing[1])+'/iteration_'+str(i)
                    os.system(cmd)  

                    saver.save(sess, './saved_models_'+str(self.name_dataset)+'/num_layers_'+str(self.num_layers)+'/dim_layers_'+str(self.dim_layers[1])+'/num_inducing_'+str(self.num_inducing[1])+'/iteration_'+str(i)+'/saved_DeepConvGP', global_step=i)  
                    print('Saved checkpoint')
    
                    #################################################################
                    #### get the final segmentations over the entire testing set ####
                    #################################################################

                    cmd = 'mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)
                    os.system(cmd)

                    if self.data_type=='2D':
                        num_testing_subjects = X_testing.shape[0]
                    elif self.data_type=='3D':
                        num_testing_subjects = len(X_testing.keys())

                    if self.data_type=='2D':

                        iterate_over_this = range(num_testing_subjects)
                    elif self.data_type=='3D':
                        iterate_over_this = X_testing.keys()

                    for _ in iterate_over_this:

                        print('*******************************')
                        print('we are at image num '+str(_))
                        print('********************************')

                        if self.data_type=='2D':

                            current_image = X_testing[_,...]
                            current_mask = masks_testing[_,...]
                            shape_of_data = X_testing[_,...].shape
                            current_label = Y_testing[_,...]
                            
                        elif self.data_type=='3D':

                            current_image = X_testing[_]
                            current_mask = masks_testing[_]
                            shape_of_data = X_testing[_].shape
                            current_label = Y_testing[_]

                        size_cube_input1 = self.dim_input//2
                        size_cube_output1 = self.dim_last_layer//2
                        size_cube_input2 = self.dim_input - size_cube_input1
                        size_cube_output2 = self.dim_last_layer - size_cube_output1
                        print(size_cube_input1)
                        print(size_cube_output1)
                        print(size_cube_input2)
                        print(size_cube_output2)

                        if self.data_type=='2D':
                                
                            patches, patches_labels, shape_of_data_after_padding, current_mask_padded, current_image_padded, current_label_padded = extract_2D_cubes_input_seg(input_image = current_image, output_image = current_label,
                                semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                                semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                                dim_output = self.dim_output, mask = current_mask)

                        elif self.data_type=='3D':

                            patches,  shape_of_data_after_padding, current_mask_padded, current_image_padded = extract_3D_cubes_input_seg_regression(input_image = current_image, 
                                semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
                                semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
                                dim_output = self.dim_output, mask = current_mask)

                        print('size of what we got from custom made non-overlapping 3D cuube extraction')
                        print(patches.shape)

                    
                        num_batch_test_time = 10
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

                            pred_mean_now, pred_var_epistemic_now, pred_var_distributional_now = sess.run([f_mean_testing, f_var_epistemic_testing, 
                                f_var_distributional_testing], feed_dict={self.X_testing:patches[lista_batches[i_batch]]})
                            print('*** size of segmentation moments ***')
                            print(pred_mean_now.shape)
                            print(pred_var_epistemic_now.shape)
                            print(pred_var_distributional_now.shape)
                            if self.data_type=='2D':

                                mean_segmentation.append(pred_mean_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_output)))
                                var_epistemic_segmentation.append(pred_var_epistemic_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_output)))
                                var_distributional_segmentation.append(pred_var_distributional_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_output)))

                            elif self.data_type=='3D':

                                mean_segmentation.append(pred_mean_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_last_layer, self.dim_output)))
                                var_epistemic_segmentation.append(pred_var_epistemic_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_last_layer, self.dim_output)))
                                var_distributional_segmentation.append(pred_var_distributional_now.reshape((-1, self.dim_last_layer, self.dim_last_layer, self.dim_last_layer, self.dim_output)))

                        mean_segmentation = np.concatenate(mean_segmentation, axis = 0)
                        var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = 0)
                        var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = 0)

                        if self.data_type=='2D':

                            mean_segmentation = [np.expand_dims(unblockshaped(arr = mean_segmentation[..., mata], h = shape_of_data_after_padding[0],
                                w = shape_of_data_after_padding[1]),axis=-1) for mata in range(self.dim_output)]
                            mean_segmentation = np.concatenate(mean_segmentation, axis = -1)

                            var_epistemic_segmentation = [np.expand_dims(unblockshaped(arr = var_epistemic_segmentation[..., mata], 
                                h = shape_of_data_after_padding[0], w = shape_of_data_after_padding[1]),
                                axis=-1) for mata in range(self.dim_output)]
                            var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = -1)

                            var_distributional_segmentation = [np.expand_dims(unblockshaped(arr = var_distributional_segmentation[..., mata], 
                                h = shape_of_data_after_padding[0], w = shape_of_data_after_padding[1]),
                                axis=-1) for mata in range(self.dim_output)]
                            var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = -1)

                            ###################################
                            ######## Manual annotation ########
           
                            cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/manual_annotation'
                            os.system(cmd)

                            print('size of current label padded')
                            print(current_label_padded.shape)

                            plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                            plt.colorbar()
                            plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/manual_annotation/simple.png')
                            plt.close()	
                    
                            for kkt_pe_bat in range(self.dim_output):
                        
                                plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                                plt.imshow(apply_mask(var_distributional_segmentation[...,kkt_pe_bat], 
                                    current_mask_padded[...,0]), cmap='magma', alpha=0.6)
                                plt.colorbar()
                                plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/manual_annotation/distributional_uncertainty_'+str(kkt_pe_bat)+'.png')
                                plt.close()	
                        
                                plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                                plt.imshow(apply_mask(var_epistemic_segmentation[...,kkt_pe_bat], 
                                    current_mask_padded[...,0]), cmap='magma', alpha=0.6)
                                plt.colorbar()
                                plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/manual_annotation/epistemic_uncertainty_'+str(kkt_pe_bat)+'.png')
                                plt.close()	

                                plt.imshow(current_label_padded[...,0], cmap='gray',alpha=0.6)
                                plt.imshow(apply_mask(mean_segmentation[...,kkt_pe_bat], 
                                    current_mask_padded[...,0]), cmap='magma', alpha=0.6)
                                plt.colorbar()
                                plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/manual_annotation/mean_segmentation_'+str(kkt_pe_bat)+'.png')
                                plt.close()	


                            ####### Raw Image ##############

                            cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/raw_image'
                            os.system(cmd)

                            plt.imshow(current_image_padded[...,0])
                            plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/raw_image/simple.png')
                            plt.close()	
                    
                            for kkt_pe_bat in range(self.dim_output):
                                    
                                plt.imshow(current_image_padded[...,0], alpha=0.6)
                                plt.imshow(apply_mask(var_distributional_segmentation[...,kkt_pe_bat], 
                                    current_mask_padded[...,0]), cmap='magma', alpha=0.6)
                                plt.colorbar()
                                plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/raw_image/distributional_uncertainty_'+str(kkt_pe_bat)+'.png')
                                plt.close()	
                        
                                plt.imshow(current_image_padded[...,0], alpha=0.6)
                                plt.imshow(apply_mask(var_epistemic_segmentation[...,kkt_pe_bat], 
                                    current_mask_padded[...,0]), cmap='magma', alpha=0.6)
                                plt.colorbar()
                                plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/raw_image/epistemic_uncertainty_'+str(kkt_pe_bat)+'.png')
                                plt.close()	

                                plt.imshow(current_image_padded[...,0], alpha=0.6)
                                plt.imshow(apply_mask(mean_segmentation[...,kkt_pe_bat], 
                                    current_mask_padded[...,0]), cmap='magma', alpha=0.6)
                                plt.colorbar()
                                plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/raw_image/mean_segmentation_'+str(kkt_pe_bat)+'.png')
                                plt.close()	
    

                            '''

                            ################################
                            ####### Mean Segmentations #####
                            ################################

                            cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/sampled_seg'
                            os.system(cmd)

                            plt.imshow(apply_mask(mean_segmentation, current_mask_padded), cmap='gray')
                            plt.colorbar()
                            plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/sampled_seg/mean_segmentation.png')
                            plt.colorbar()
                            plt.close()	

                            plt.imshow(apply_mask(predictive_entropy, current_mask_padded), cmap='gray')
                            plt.colorbar()
                            plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/sampled_seg/predictive_entropy.png')
                            plt.colorbar()
                            plt.close()	

                            ############################################
                            ####### Distributional Uncertainty #########
                            ############################################

                            cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/uncertainty'
                            os.system(cmd)
                    
                            plt.imshow(apply_mask(var_distributional_segmentation, current_mask_padded), cmap='magma')
                            plt.colorbar()
                            plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/uncertainty/distributional_uncertainty.png')
                            plt.close()	
                            
                            plt.imshow(apply_mask(var_epistemic_segmentation, current_mask_padded), cmap='magma')
                            plt.colorbar()
                            plt.savefig('./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/uncertainty/epistemic_uncertainty.png')
                            plt.close()	                

                            '''

                        elif self.data_type=='3D':

                            mean_segmentation = [np.expand_dims(uncubify(mean_segmentation[...,kkt], 
                                (shape_of_data_after_padding[0], shape_of_data_after_padding[1], 
                                shape_of_data_after_padding[2])),axis=-1) for kkt in range(self.dim_output)]
                            mean_segmentation = np.concatenate(mean_segmentation, axis = -1)

                            var_epistemic_segmentation = [np.expand_dims(uncubify(var_epistemic_segmentation[...,kkt], 
                                (shape_of_data_after_padding[0], shape_of_data_after_padding[1], 
                                shape_of_data_after_padding[2])),axis=-1) for kkt in range(self.dim_output) ]
                            var_epistemic_segmentation = np.concatenate(var_epistemic_segmentation, axis = -1)
                            
                            var_distributional_segmentation = [np.expand_dims(uncubify(var_distributional_segmentation[...,kkt], 
                                (shape_of_data_after_padding[0], shape_of_data_after_padding[1], 
                                shape_of_data_after_padding[2])),axis=-1) for kkt in range(self.dim_output)]
                            var_distributional_segmentation = np.concatenate(var_distributional_segmentation, axis = -1)
                            
                            current_mask_padded = np.reshape(current_mask_padded, 
                                (shape_of_data_after_padding[0], shape_of_data_after_padding[1], shape_of_data_after_padding[2]))

                            ### TODO -- need to mask the above in the final nifti files ###
                    
                            cmd='mkdir -p ./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)
                            os.system(cmd)

                            img = nib.Nifti1Image(mean_segmentation, self.affine)
                            nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/mean_segmentation.nii.gz' )
                    
                            img = nib.Nifti1Image(var_epistemic_segmentation, self.affine)
                            nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/epistemic_var_segmentation.nii.gz' )
                    
                            img = nib.Nifti1Image(var_distributional_segmentation, self.affine)
                            nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/distributional_var_segmentation.nii.gz' )
  
                            img = nib.Nifti1Image(current_image_padded, self.affine)
                            nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/input.nii.gz' )

                            img = nib.Nifti1Image(mean_segmentation-current_label, self.affine)
                            nib.save(img,'./whole_image_segmentations_'+str(self.name_dataset)+'/iteration_'+str(i)+'/image_num_'+str(_)+'/brain_pad.nii.gz' )
          

                print('at iteration '+str(i) + ' we have nll : '+str(costul_actual) + 're cost :'+str(re_cost_actual)+' kl cost :'+str(kl_cost_actual))



