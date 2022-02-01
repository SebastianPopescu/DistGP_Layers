# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from DGP_activated_Segmentation_Classification import Dist_GP_activated_NN
import argparse
from init_variables import *
import skimage.io
import os
from sklearn.cluster import  KMeans
from sklearn.feature_extraction.image import extract_patches_2d
from loading_data import *
import nibabel as nib
from data_processing import cubify
from sklearn.model_selection import KFold

def one_hot_encoder(input, num_classes):

    ### Usually 0 is the background which is ignored because of masking

    object = np.zeros(shape=(input.shape[0],num_classes))
    for i in range(input.shape[0]):
        object[i,int(input[i])-1] = 1.0
    return object	

def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret


def absoluteFilePaths(dirpath):
   
    ### dirpath has to be absolute path ####

    list_realpaths = []
    filenames = os.listdir(dirpath)

    for f in filenames:
        list_realpaths.append(os.path.abspath(os.path.join(dirpath, f)))

    return list_realpaths

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type = int, help = 'the number of hidden layers + classification layer')
    parser.add_argument('--dim_filters', type = int, help = 'dimension of filters for hidden layers')
    parser.add_argument('--dim_layers', type = int, help = 'the number of channels to use for hidden layers-- GP part')
    parser.add_argument('--dim_convolutionally_warped_gp', type = int, help = 'the number of channels to warp the GP')
    parser.add_argument('--num_inducing', type = int, help = 'the number of inducing points at hidden layers')
    parser.add_argument('--num_iterations', type = int, help = 'number of training iterations' )
    parser.add_argument('--path_input_scans', type = str, help = 'absolute path of folder that holds the input scans')
    parser.add_argument('--path_output_scans', type = str, help = 'absolute path of folder that holds the output scans/segmentation')
    parser.add_argument('--path_example_scan', type = str, help = 'absolute path of example scan')
    parser.add_argument('--dataset_name', type = str, help = 'dataset name')
    parser.add_argument('--dim_output', type = str, help = 'number of classes to segment')
    args = parser.parse_args()

    num_inducing_first_layer = 250

    ########## Masks ############
    
    #list_realpath_masks = absoluteFilePaths(dirpath = '/vol/biomedic3/sgp15/CamCan/T2w/masks')
    #list_realpath_masks = sorted(list_realpath_masks)
    
    ##### Input data ############
    
    list_realpath_input = absoluteFilePaths(dirpath = args.path_input_scans)
    list_realpath_input = sorted(list_realpath_input)
    
    ####### Segmentations ##########
    
    list_realpath_segmentations = absoluteFilePaths(dirpath = args.path_output_scans)
    list_realpath_segmentations = sorted(list_realpath_segmentations)
  
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    cv_num = 0
    control = 0

    for train_index, test_index in kf.split(range(len(list_realpath_segmentations))):

        if control==cv_num:

            #### Masks #####
            #list_realpath_masks_training =  parse_string_list(string_list = list_realpath_masks, index = train_index)
            #list_realpath_masks_testing =  parse_string_list(string_list = list_realpath_masks, index = test_index)
    
            ##### Input data ############
            list_realpath_input_training =  parse_string_list(string_list = list_realpath_input, index = train_index)
            list_realpath_input_testing =  parse_string_list(string_list = list_realpath_input, index = test_index)
        
            ####### Segmentations ##########
            list_realpath_segmentations_training =  parse_string_list(string_list = list_realpath_segmentations, index = train_index)
            list_realpath_segmentations_testing =  parse_string_list(string_list = list_realpath_segmentations, index = test_index)

            control+=1

        else:

            control+=1

    #dict_X_training = data_factory_whole_brain(list_of_nifty_files = list_realpath_input_training)
    dict_X_testing = data_factory_whole_brain(list_of_nifty_files = list_realpath_input_testing[:10])

    #dict_X_training_masks = data_factory_whole_brain(list_of_nifty_files = list_realpath_masks_training)
    #dict_X_testing_masks = data_factory_whole_brain(list_of_nifty_files = list_realpath_masks_testing)

    #dict_Y_training = data_factory_whole_brain(list_of_nifty_files = list_realpath_segmentations_training)
    dict_Y_testing = data_factory_whole_brain(list_of_nifty_files = list_realpath_segmentations_testing[:10])
        
    #################################
    ### Preprocess Raw images #######
    #################################

    '''
    for key in dict_X_training.keys():
        print(key)
        dict_X_training[key] = normalise_one_one(dict_X_training[key])
        dict_X_training[key][dict_X_training[key]==-1.0] = -3.5
    '''


    for key in dict_X_testing.keys():
        print(key)
        dict_X_testing[key] = normalise_one_one(dict_X_testing[key])
        dict_X_testing[key][dict_X_testing[key]==-1.0] = -3.5

    ####### Get the masks in boolean form #########

    #############################
    ########## Masks ############
    #############################
    '''
    dict_X_training_masks = defaultdict()
    for key in dict_X_training.keys():
        print(key)
        dict_X_training_masks[key] = np.array(dict_Y_training[key]>0.0, dtype=np.bool)
    '''


    dict_X_testing_masks = defaultdict()
    for key in dict_X_testing.keys():
        print(key)
        dict_X_testing_masks[key] = np.array(dict_Y_testing[key]>0.0, dtype=np.bool)
        

    patches = []

    for _ in range(5):

        developement_set = dict_X_testing[_]
        print(developement_set.shape)
        print('**************************************')
        patches.append(cubify(developement_set[:195:,:230,:185], [5,5,5]))
    
    patches = np.concatenate(patches, axis = 0)
    
    print(patches.shape)
    patches = patches.reshape((-1, 5**3))
    np.random.shuffle(patches)
    patches = patches[:5000,...]

    km = KMeans(n_clusters = num_inducing_first_layer).fit(patches)
    init_patches = km.cluster_centers_
    init_patches = init_patches.astype(np.float32)
    print(init_patches.shape)

    ##############################################
    ##### crate lists of model configuration #####
    ##############################################

    dim_output = args.dim_output

    dim_layers = [1]
    dim_layers.extend([args.dim_layers for _ in range(args.num_layers-1)])
    dim_layers.append(dim_output)

    dim_convolutionally_warped_gp = [args.dim_convolutionally_warped_gp for _ in range(args.num_layers-1)]

    dim_filters = [5]
    dim_filters.extend([args.dim_filters for _ in range(args.num_layers-1)])
    
    num_strides = [1]
    num_strides.extend([1 for _ in range(args.num_layers-1)])

    num_inducing = [num_inducing_first_layer]
    num_inducing.extend([args.num_inducing for _ in range(args.num_layers-1)])

    ### get affine ###
    nvm = nib.load(args.example_scan)
 
    obiect = Dist_GP_activated_NN(num_data = None, num_test = len(dict_X_testing.keys()),
        dim_input = 35, dim_output = dim_output, 
        num_iterations = args.num_iterations, num_layers = args.num_layers, 
        dim_layers = dim_layers, dim_convolutionally_warped_gp = dim_convolutionally_warped_gp,
        num_inducing = num_inducing, type_var = 'full', 
        num_batch = 4, num_samples_testing = 10, 
        dim_filters = dim_filters, num_strides = num_strides, 
        name_dataset = args.dataset_name, import_model = True, iteration_restored = 50000, learning_rate = 1e-4,
        dim_input_channels = 1, num_patches_per_subject = 1,
        num_subjects = 2, num_averaged_gradients = 2, use_bayesian_deep_supervision = True, testing_time = True,
        data_type = '3D', name_dataset_to_import = args.dataset_name, affine = nvm.affine, multi_kernel = False,
        X_training = None, Y_training = None, 
        masks_training = None, X_testing = dict_X_testing, Y_testing = dict_Y_testing, 
        masks_testing = dict_X_testing_masks, Z_init = init_patches, use_masks=False)

