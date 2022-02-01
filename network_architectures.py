# -*- coding: utf-8 -*-
import numpy as np
from model_details import model_details
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
DTYPE=tf.float32

class network_architectures(model_details):

    def __init__(self, **kwargs):

        model_details.__init__(self, **kwargs)


    ###########################################
    ######### Baseline DeepMedic model ########
    ###########################################

    def baseline_deep_medic(self, inputul, propagate_layers_object, conditional_GP_object, training_time, full_cov=False):

        ###################################################################################################
        #### this is giving just the representation learning embedding of the final segmentation block ####
        ###################################################################################################

        #### First layer is Standard GP propagated in a convolutional manner ####
        list_inputul_mean = []
        list_inputul_var = []

        if training_time:
            list_kl_term = []
        l = 1
        output_now = propagate_layers_object.propagate_euclidean_CNN(l = l, input_prev_layer = inputul,
            conditional_GP_object = conditional_GP_object, training_time = training_time,
            layer_type = 'CNN', full_cov=False)
        
        inputul_mean = output_now[0]
        inputul_var_epistemic = output_now[1]
        inputul_var_distibutional = output_now[2]
        inputul_var = inputul_var_epistemic + inputul_var_distibutional
        if training_time:
            kl_term = output_now[-1]
            list_kl_term.append(kl_term)
        list_inputul_mean.append(inputul_mean)
        list_inputul_var.append(inputul_var)


        #### Subsequent layers are in Wasserstein Space ####

        for l in range(2, self.num_layers):

            output_now = propagate_layers_object.propagate_wasserstein_CNN(l = l, X_mean = inputul_mean, X_var = inputul_var,
                conditional_GP_object = conditional_GP_object, training_time = training_time,
                layer_type = 'CNN', full_cov=False)
            inputul_mean = output_now[0]
            inputul_var_epistemic = output_now[1]
            inputul_var_distibutional = output_now[2]
            inputul_var = inputul_var_epistemic + inputul_var_distibutional
            if training_time:
                kl_term = output_now[-1]
                list_kl_term.append(kl_term)

            list_inputul_mean.append(inputul_mean)
            list_inputul_var.append(inputul_var)


        l = self.num_layers

        output_now = propagate_layers_object.propagate_wasserstein_CNN_last_layer(l = l, X_mean = inputul_mean, X_var = inputul_var,
            conditional_GP_object = conditional_GP_object, training_time = training_time,
            layer_type = 'CNN', full_cov=False)
        inputul_mean = output_now[0]
        inputul_var_epistemic = output_now[1]
        inputul_var_distibutional = output_now[2]
        inputul_var = inputul_var_epistemic + inputul_var_distibutional

        if training_time:
            kl_term = output_now[-1]
            list_kl_term.append(kl_term)

        list_inputul_mean.append(inputul_mean)
        list_inputul_var.append(inputul_var)

        ##########################################################################
        #### returns list of moments to be used for Bayesian Deep Supervision ####
        ##########################################################################
        if training_time:
            return list_inputul_mean, list_inputul_var, tf.reduce_sum(list_kl_term)
        else:
            return list_inputul_mean, list_inputul_var

    def baseline_deep_medic_uncertainty_decomposed(self, inputul, propagate_layers_object, conditional_GP_object, full_cov=False):

        ###################################################################################################
        #### this is giving just the representation learning embedding of the final segmentation block ####
        ###################################################################################################

        list_f_mean = []
        list_f_var_epistemic = []
        list_f_var_distributional = []

        #### First layer is Standard GP propagated in a convolutional manner ####

        l = 1
        output_now = propagate_layers_object.propagate_euclidean_CNN(l = l, input_prev_layer = inputul,
            conditional_GP_object = conditional_GP_object, training_time = False,
            layer_type = 'CNN', full_cov=False)
        
        inputul_mean = output_now[0]
        inputul_var_epistemic = output_now[1]
        inputul_var_distibutional = output_now[2]
        inputul_var = inputul_var_epistemic + inputul_var_distibutional
        list_f_mean.append(inputul_mean)
        list_f_var_epistemic.append(inputul_var_epistemic)
        list_f_var_distributional.append(inputul_var_distibutional)

        #### Subsequent layers are in Wasserstein Space ####

        for l in range(2, self.num_layers):

            output_now = propagate_layers_object.propagate_wasserstein_CNN(l = l, X_mean = inputul_mean, X_var = inputul_var,
                conditional_GP_object = conditional_GP_object, training_time = False,
                layer_type = 'CNN', full_cov=False)
            inputul_mean = output_now[0]
            inputul_var_epistemic = output_now[1]
            inputul_var_distibutional = output_now[2]
            inputul_var = inputul_var_epistemic + inputul_var_distibutional

            list_f_mean.append(inputul_mean)
            list_f_var_epistemic.append(inputul_var_epistemic)
            list_f_var_distributional.append(inputul_var_distibutional)


        l = self.num_layers

        output_now = propagate_layers_object.propagate_wasserstein_CNN_last_layer(l = l, X_mean = inputul_mean, X_var = inputul_var,
            conditional_GP_object = conditional_GP_object, training_time = False,
            layer_type = 'CNN', full_cov=False)
        inputul_mean = output_now[0]
        inputul_var_epistemic = output_now[1]
        inputul_var_distibutional = output_now[2]
        inputul_var = inputul_var_epistemic + inputul_var_distibutional

        list_f_mean.append(inputul_mean)
        list_f_var_epistemic.append(inputul_var_epistemic)
        list_f_var_distributional.append(inputul_var_distibutional)

        return list_f_mean, list_f_var_epistemic, list_f_var_distributional


















###########################
########## U-NET ##########
###########################

### TODO -- implement this
