import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import defaultdict
import random
from sklearn import preprocessing
import os


##################################################################
################# Helper functions ###############################
##################################################################

def check_segmentation_block(current_segmentation_block):

    ### current_segmentation_block -- shape (height, width, depth, dim_output)
    current_segmentation_block_summed = np.sum(current_segmentation_block, axis=-1,keepdims=True)
    ### current_segmentation_block_summed -- shape (height, width, depth, 1)    

    ### get background class

    current_background_class = current_segmentation_block[...,0]
    current_background_class = np.expand_dims(current_background_class, axis =-1) 
    current_tissue_classes = current_segmentation_block[...,1:]

    current_background_class[current_segmentation_block_summed==0.0] = 1.0

    ### get them together again

    final_seg_block = np.concatenate((current_background_class, current_tissue_classes),axis =-1)
  


    return final_seg_block


def apply_mask(array, mask):

    ##### array --- (dim1, dim2, dim3)
    ##### mask --- (dim1, dim2, dim3)
    ##### masks the array so that the non-ROI part is zero ####

    array[mask < 1.0] = 0.0

    return array

def blockshaped(arr, nrows, ncols):

    ############################################################    
    #### Used to get non-overlapping patches from an image #####
    ############################################################

    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):

    ###############################################################
    #### Used to re-arrange 2D patches into the original iamge ####
    ###############################################################

    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
        .swapaxes(1,2)
        .reshape(h, w))

def output_transformation(inputul):

    inputul = np.round(inputul)

    return inputul

def one_hot_encoder(input,dim_output,list_values):

    dictionar=defaultdict()
    for value,control in zip(list_values,np.arange(dim_output)):
        dictionar[value] = control

    object = np.zeros(shape=(input.shape[0],dim_output))
    for i in range(input.shape[0]):

        object[i,dictionar[int(input[i,0])]] = 1.0

    return object	

#################################################
##### Data processing for 2D Segmentation #######
#################################################

def crop_2D_block(image, central_points, semi_block_size1, semi_block_size2):

    #### basically crops a small 2D cube from a bigger 2D object including the different input channels ####

    ### image -- shape (height, width, channels)
    ### central_points -- (c1,c2)
    ### semi_block_size -- (l1,l2)

    plm = image[central_points[0]-semi_block_size1:central_points[0]+semi_block_size2,
        central_points[1]-semi_block_size1:central_points[1]+semi_block_size2,:]


    return plm

def check_and_add_zero_padding_2d_image(input_image, output_image, mask_image, central_points, semi_block_size1, semi_block_size2):

    #### checks if extracting a patch need padding or not 
    #### accounts for the case where the central_points are close to the boundary of the image and expands it with the minimum of the image

    ### image -- shape (height, width, channels)
    ### central_points -- (c1,c2)
    ### semi_block_size -- (l1,l2)

    current_shape = input_image.shape 
    min_value_image = np.min(input_image)
    padding_dimensions = []
    control=0				

    for _ in range(2):

        dim_list = []

        if central_points[_]-semi_block_size1 < 0:			
            dim_list.append(np.abs(central_points[_]-semi_block_size1))
            control+=1

        else:

            dim_list.append(0)

        if central_points[_]+semi_block_size2 > current_shape[_]:
            dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]))
            control+=1
        else:
            dim_list.append(0)

        padding_dimensions.append(tuple(dim_list))


    if control > 0:

        padding_dimensions = tuple(padding_dimensions)
        padding_dimensions_extra = list(padding_dimensions)
        padding_dimensions_extra.append(tuple([0,0]))
        padding_dimensions_extra = tuple(padding_dimensions_extra)

        input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)
        output_image = np.pad(output_image, padding_dimensions_extra, mode='constant')
        mask_image = np.pad(mask_image, padding_dimensions_extra, mode='constant')
        central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(2)]

    return input_image, output_image, mask_image, central_points

##################################################################
##################################################################
################# Training time patch extraction #################
##################################################################
##################################################################

def extract_2d_blocks_training(inputul, outputul, iteration, block_size_input, block_size_output, dim_output, num_subjects,
    num_patches_per_subject, masks, list_dim_last_layer, use_masks):

    ## inputul -- shape (num_batch, width, height, num_channels_input)
    ## outputul -- shape (num_batch, width, height, 1)
    ## masks -- shape (num_batch, width, height, 1) ### ones and zeros basically 

    #### this will extract num_subjects * num_patches_per_subject #######
    lista = np.arange(inputul.shape[0])
    np.random.seed(iteration)
    np.random.shuffle(lista)
    current_index = lista[:num_subjects]

    semi_block_size_input = int(block_size_input//2)
    semi_block_size_input2 = block_size_input - semi_block_size_input
    semi_block_size_output = int(block_size_output//2)
    semi_block_size_output2 = block_size_output - semi_block_size_output
    
    list_blocks_input = []
    list_blocks_segmentation = []
    if use_masks:
        list_blocks_masks = []
    
    ##########################################
    ####### Bayesian Deep Supervision ########
    ##########################################

    dict_bds_blocks_segmentation = defaultdict()
    if use_masks:
       dict_bds_blocks_masks = defaultdict()
    
    dict_semi_block_size_output = defaultdict()
    dict_semi_block_size_output2 = defaultdict()
    
    for _ in range(len(list_dim_last_layer)):
    
        dict_bds_blocks_segmentation[_+1] = []
        if use_masks:
            dict_bds_blocks_masks[_+1] = []      
        dict_semi_block_size_output[_+1] = int(list_dim_last_layer[_]//2)
        dict_semi_block_size_output2[_+1] = list_dim_last_layer[_] - dict_semi_block_size_output[_+1]
        
    for _ in current_index:

        ##### iterating over 2D images ###############################
        ### pad current input and output images to avoid problems ####

        if dim_output==1:

            current_input = inputul[_,...]
            current_output = outputul[_,...]
            if use_masks:
                current_mask = masks[_,...]
            #### shape of current image ####
            current_shape = inputul[_,...].shape
        else:

            current_input = inputul[_,...]
            current_input = np.expand_dims(current_input, axis =-1)
            current_shape = inputul[_].shape
            current_output = outputul[_,...]
            ### need to expand the dimension of the output to match the routine
            current_output = np.expand_dims(current_output, axis =-1)
            ### need to perform the one hot encoding here 
            current_output = one_hot_encoder(input = np.reshape(current_output,(-1,1)),
                dim_output = dim_output, list_values= [kkt for kkt in range(1,dim_output+1)])
            current_output = np.reshape(current_output, (current_shape[0], current_shape[1], dim_output))
            if use_masks:
                current_mask = masks[_]
                current_mask = np.expand_dims(current_mask, axis =-1)


        '''
        #################################################################################################################
        #### random places being extracted -- most likely not containing any segmentation besides background class ######
        #################################################################################################################

        list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), num_patches_per_subject)
        list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), num_patches_per_subject)

        for __ in range(num_patches_per_subject):
            
            #### iterate over the 2 locations of the 3D cubes #####
            central_points = [list_of_random_places1[__], list_of_random_places2[__]]

            current_input_padded, current_output_padded, current_mask_padded, central_points = check_and_add_zero_padding_2d_image(current_input, 
                current_output, current_mask, central_points, semi_block_size_input, semi_block_size_input2)

            list_blocks_segmentation.append(crop_2D_block(current_output_padded, central_points,
                semi_block_size_output, semi_block_size_output2))
            list_blocks_input.append(crop_2D_block(current_input_padded, central_points,
                semi_block_size_input, semi_block_size_input2))
            list_blocks_masks.append(crop_2D_block(current_mask_padded, central_points,
                semi_block_size_output, semi_block_size_output2))


        '''
        ###############################################################################################
        ##### specifically extract 2D patches with a non-background class #############################
        ###############################################################################################
        if dim_output==1:
            kkt=dim_output+1
        else:
            kkt=dim_output


        for class_number in range(1,kkt):

            ####################################
            ##### Class number class_number ####
            ####################################

            if dim_output==1:


                indices_class = np.where(current_output[...,0] == class_number)
            else:
                indices_class = np.where(current_output[...,class_number] == 1)

            indices_class_dim1 = indices_class[0]
            indices_class_dim2 = indices_class[1]

            if len(indices_class_dim1)==0:

                print('class '+str(class_number)+' not found in current image')

            else:
                        
                list_of_random_places = random.sample(range(0,len(indices_class_dim1)), num_patches_per_subject)

                for __ in range(num_patches_per_subject):

                    central_points = [indices_class_dim1[list_of_random_places[__]],
                        indices_class_dim2[list_of_random_places[__]]]

                    current_input_padded, current_output_padded, current_mask_padded, central_points = check_and_add_zero_padding_2d_image(current_input,
                        current_output, current_mask, central_points, semi_block_size_input, semi_block_size_input2)

                    for plm_bds in range(len(list_dim_last_layer)):
        
                        dict_bds_blocks_segmentation[plm_bds+1].append(crop_2D_block(current_output_padded, central_points,
                            dict_semi_block_size_output[plm_bds+1],dict_semi_block_size_output2[plm_bds+1]))
                        if use_masks:
                            dict_bds_blocks_masks[plm_bds+1].append(crop_2D_block(current_mask_padded, central_points,
                                dict_semi_block_size_output[plm_bds+1],dict_semi_block_size_output2[plm_bds+1]))

                    list_blocks_segmentation.append(crop_2D_block(current_output_padded, central_points,
                        semi_block_size_output,semi_block_size_output2))
                    list_blocks_input.append(crop_2D_block(current_input_padded, central_points,
                        semi_block_size_input,semi_block_size_input2))
                    if use_masks:
                        list_blocks_masks.append(crop_2D_block(current_mask_padded, central_points,
                            semi_block_size_output,semi_block_size_output2))

    list_blocks_input = np.stack(list_blocks_input)
    list_blocks_segmentation = np.stack(list_blocks_segmentation)
    if use_masks:
        list_blocks_masks  = np.stack(list_blocks_masks)

    for plm_bds in range(len(list_dim_last_layer)):

        dict_bds_blocks_segmentation[plm_bds+1] = np.stack(dict_bds_blocks_segmentation[plm_bds+1])
        if use_masks:
            dict_bds_blocks_masks[plm_bds+1] = np.stack(dict_bds_blocks_masks[plm_bds+1])

        shape_of_seg = dict_bds_blocks_segmentation[plm_bds+1].shape
        dict_bds_blocks_segmentation[plm_bds+1] = dict_bds_blocks_segmentation[plm_bds+1].reshape((-1,1))
        dict_bds_blocks_segmentation[plm_bds+1] = dict_bds_blocks_segmentation[plm_bds+1].reshape((shape_of_seg[0],
            shape_of_seg[1],shape_of_seg[2],dim_output))

    shape_of_seg = list_blocks_segmentation.shape
    list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
    #list_blocks_segmentation = output_transformation(list_blocks_segmentation)
    #enc = preprocessing.OneHotEncoder()
    #enc.fit(list_blocks_segmentation)
    #list_blocks_segmentation = enc.transform(list_blocks_segmentation).toarray()
    #list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
    #list_blocks_segmentation = OneHotEncoder(list_blocks_segmentation)
    #list_blocks_segmentation = one_hot_encoder(input = list_blocks_segmentation,dim_output = 2, list_values = [0.0,1.0])

    list_blocks_segmentation = list_blocks_segmentation.reshape((shape_of_seg[0],shape_of_seg[1],shape_of_seg[2],dim_output))

    output_bds_seg = []
    if use_masks:
        bds_masks = []
    for key in dict_bds_blocks_segmentation.keys():
        output_bds_seg.append(dict_bds_blocks_segmentation[key])
        if use_masks:
            bds_masks.append(dict_bds_blocks_masks[key])

    if use_masks:
        return list_blocks_input, list_blocks_segmentation, list_blocks_masks, output_bds_seg, bds_masks
    else:
        return list_blocks_input, list_blocks_segmentation, output_bds_seg

##################################################################
##################################################################
################# Testing time patch extraction ##################
##################################################################
##################################################################

######  Extract non-overlapping 2D patches in segmentation space #############
###### also extracts the overlapping bigger 2D patches in raw input space ####

def extract_2D_cubes_input_seg(input_image, output_image, semi_block_size_input1, semi_block_size_output1,
    semi_block_size_input2, semi_block_size_output2, dim_output, mask, use_masks):

    if dim_output==1:        
        pass

    else:

        current_shape = output_image.shape
        ### need to expand the dimension of the output to match the routine
        output_image = np.expand_dims(output_image, axis =-1)
        ### need to perform the one hot encoding here 
        output_image = one_hot_encoder(input = np.reshape(output_image,(-1,1)),
            dim_output = dim_output, list_values= [kkt for kkt in range(1,dim_output+1)])
        output_image = np.reshape(output_image, (current_shape[0], current_shape[1], dim_output))

        if use_masks:
            mask = np.expand_dims(mask, axis =-1)
        input_image = np.expand_dims(input_image, axis = -1)

  
    block_size_output = semi_block_size_output1 + semi_block_size_output2
    block_size_input = semi_block_size_input1 + semi_block_size_input2
    diff_semi_block = np.abs(semi_block_size_output1 - semi_block_size_input1)


    shape_of_data = output_image.shape

    ### dimension 1 ###
    diff_dim1 = shape_of_data[0] %  block_size_output
    if diff_dim1!=0:
        diff_dim1 = block_size_output - diff_dim1
 
    ### dimension 2 ###
    diff_dim2 = shape_of_data[1] % block_size_output
    if diff_dim2!=0:
        diff_dim2 = block_size_output - diff_dim2


    #####################################################################
    ### pad output space so that it is divisible by block_size_output ###
    #####################################################################
    print('***************')
    print(diff_dim1)
    print(diff_dim2)
    print(input_image.shape)
    print(output_image.shape)

    output_image = np.pad(array = output_image, pad_width = ((diff_dim1,0), (diff_dim2,0), (0,0)), mode='constant')
    if use_masks:
        mask = np.pad(array = mask, pad_width = ((diff_dim1,0), (diff_dim2,0), (0,0)), mode='constant')
    input_image = np.pad(array = input_image, pad_width = ((diff_dim1,0), (diff_dim2,0), (0,0)), mode='constant')

    

    ###################################################################################################################################
    ###### Remainder -- to get from output_image_coordinates to input_image_coordinates add on all dimensions + diff_semi_block #######
    ###################################################################################################################################


    ######  Extract non-overlapping 3D cubes in Regression output space ###############################################
    ###### also extracts the overlapping bigger 3D cubes in raw input space ###########################################
    ###### extracts non-overlapping 3D blocks contrained by an ROI box  ###############################################

    shape_of_input_data = input_image.shape
    shape_of_output_data = output_image.shape

    num_cubes_dim1 = np.int(shape_of_output_data[0] // block_size_output)
    num_cubes_dim2 = np.int(shape_of_output_data[1] // block_size_output)
    

    list_input_cubes = []
    list_output_cubes = []

    for i in range(num_cubes_dim1):
        for j in range(num_cubes_dim2):

            ##################################################
            ### extract segmentation output space 2D patch ###
            ##################################################

            list_output_cubes.append(output_image[block_size_output*i:(block_size_output*(i+1)),
                block_size_output*j:(block_size_output*(j+1) ),:]  )
    
            ########################################
            ### extract raw input space 2D patch ###
            ########################################

            #### it might require some additional zero-padding if near the margins of the image #####

            list_input_cubes.append(check_and_add_zero_padding_2d_image_test_time(input_image = input_image, output_image = output_image,
                lower_points = [block_size_output*i, block_size_output*j],
                upper_points = [block_size_output*(i+1), block_size_output*(j+1)],
                diff_semi_block_size1 = diff_semi_block, diff_semi_block_size2 = diff_semi_block))
    

    list_output_cubes = np.stack(list_output_cubes)
    list_input_cubes = np.stack(list_input_cubes)

    if use_masks:
        return list_input_cubes, list_output_cubes, shape_of_output_data, mask, output_image, input_image
    else:
        return list_input_cubes, list_output_cubes, shape_of_output_data, output_image, input_image

def check_and_add_zero_padding_2d_image_test_time(input_image, output_image,
    lower_points, upper_points, diff_semi_block_size1, diff_semi_block_size2):

    #### checks if extracting a patch need padding or not 
    #### accounts for the case where the central_points are close to the boundary of the image and expands it with the minimum of the image

    ### image -- shape (height, width, channels)
    ### lower_points -- (c1, c2)
    ### upper_points -- (c1, c2)
    ### semi_block_size -- (l1, l2)

    current_shape = input_image.shape 
    min_value_image = np.min(input_image)
    padding_dimensions = []
    control=0				

    for _ in range(2):

        dim_list = []

        if lower_points[_] - diff_semi_block_size1 < 0:			
            dim_list.append(np.abs(lower_points[_]-diff_semi_block_size1))
            control+=1

        else:

            dim_list.append(0)

        if upper_points[_] + diff_semi_block_size2 > current_shape[_]:
            dim_list.append(np.abs(upper_points[_]+diff_semi_block_size2 - current_shape[_]))
            control+=1
        else:
            dim_list.append(0)

        padding_dimensions.append(tuple(dim_list))

    if control > 0:

        padding_dimensions = tuple(padding_dimensions)
        padding_dimensions_extra = list(padding_dimensions)
        padding_dimensions_extra.append(tuple([0,0]))
        padding_dimensions_extra = tuple(padding_dimensions_extra)
        
    
        ###################################
        ##### crop as much as possible ####
        ###################################
        
        c1 = np.abs(padding_dimensions_extra[0][0]-diff_semi_block_size1)
        c2 = np.abs(padding_dimensions_extra[0][1]-diff_semi_block_size1)
        c3 = np.abs(padding_dimensions_extra[1][0]-diff_semi_block_size1)
        c4 = np.abs(padding_dimensions_extra[1][1]-diff_semi_block_size1)

        patch = input_image[lower_points[0]- c1: upper_points[0]+c2,
            lower_points[1]-c3 : upper_points[1]+c4,:]

        #######################################
        ##### afterwards pad the remainder ####
        #######################################

        patch = np.pad(patch, padding_dimensions_extra, mode='constant',
            constant_values = min_value_image)

    else:

        patch = input_image[lower_points[0]-diff_semi_block_size1 : upper_points[0]+diff_semi_block_size1,
            lower_points[1]-diff_semi_block_size2 : upper_points[1]+diff_semi_block_size2,:]

    return patch

##################################################################
####### This is for splitting 3D objects into 3d cubes ###########
##################################################################

###############################################
#### non-overlapping cubes from 3D block ######
###############################################

def cubify(arr, newshape):

    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

###################################################
#### gather small cubes into bigger 3D block ######
###################################################

def uncubify(arr, oldshape):

    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def crop_3D_block(image, central_points, semi_block_size1, semi_block_size2):


    #### basically crops a small 3D cube from a bigger 3D object

    ### image -- shape (height, width, depth, channels)
    ### central_points -- (c1,c2,c3)
    ### semi_block_size -- (l1,l2,l3)

    plm = image[central_points[0]-semi_block_size1:central_points[0]+semi_block_size2,
        central_points[1]-semi_block_size1:central_points[1]+semi_block_size2,
        central_points[2]-semi_block_size1:central_points[2]+semi_block_size2,:]

    return plm

def check_and_add_zero_padding_3d_image(input_image, output_image, mask_image, central_points,
    semi_block_size1, semi_block_size2, use_masks):

    #### checks if extracting a block need padding or not 
    #### accounts for the case where the central_points are close to the boundary of the brain scan and expands it with 0s

    ### image -- shape (height, width, depth, channels)
    ### central_points -- (c1,c2,c3)
    ### semi_block_size -- (l1,l2,l3)

    current_shape = input_image.shape 
    min_value_image = np.min(input_image)
    padding_dimensions = []
    control=0				

    for _ in range(3):

        dim_list = []

        if central_points[_]-semi_block_size1 < 0:			
            dim_list.append(np.abs(central_points[_]-semi_block_size1))
            control+=1

        else:

            dim_list.append(0)

        if central_points[_]+semi_block_size2 > current_shape[_]:
            dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]))
            control+=1
        else:
            dim_list.append(0)

        padding_dimensions.append(tuple(dim_list))

    if control > 0:

        padding_dimensions = tuple(padding_dimensions)
        padding_dimensions_extra = list(padding_dimensions)
        padding_dimensions_extra.append(tuple([0,0]))
        padding_dimensions_extra = tuple(padding_dimensions_extra)


        input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)
 
        if use_masks:
            mask_image = np.pad(mask_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)

        output_image = np.pad(output_image, padding_dimensions_extra, mode='constant')
        central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]

    if use_masks:
        return input_image, output_image, mask_image, central_points
    else:
        return input_image, output_image, central_points

def extract_3d_blocks_training(inputul, outputul, iteration, block_size_input, block_size_output, dim_output, num_subjects,
    num_patches_per_subject, masks, list_dim_last_layer, use_masks):


    ## inputul -- shape (num_batch, width, height, num_channels_input)
    ## outputul -- shape (num_batch, width, height, 1)
    ## masks -- shape (num_batch, width, height, 1) ### ones and zeros basically 

    #### this will extract num_subjects * num_patches_per_subject #######

    lista = np.arange(len(inputul.keys()))
    np.random.seed(iteration)
    np.random.shuffle(lista)
    current_index = lista[:num_subjects]

    semi_block_size_input = int(block_size_input//2)
    semi_block_size_input2 = block_size_input - semi_block_size_input
    semi_block_size_output = int(block_size_output//2)
    semi_block_size_output2 = block_size_output - semi_block_size_output
    
    list_blocks_input = []
    list_blocks_segmentation = []
    if use_masks:
        list_blocks_masks = []
    
    ##########################################
    ####### Bayesian Deep Supervision ########
    ##########################################

    dict_bds_blocks_segmentation = defaultdict()
    if use_masks:
        dict_bds_blocks_masks = defaultdict()
    
    dict_semi_block_size_output = defaultdict()
    dict_semi_block_size_output2 = defaultdict()
    
    for _ in range(len(list_dim_last_layer)):
    
        dict_bds_blocks_segmentation[_+1] = []
        if use_masks:
            dict_bds_blocks_masks[_+1] = []      
        dict_semi_block_size_output[_+1] = int(list_dim_last_layer[_]//2)
        dict_semi_block_size_output2[_+1] = list_dim_last_layer[_] - dict_semi_block_size_output[_+1]
        
    for _ in current_index:

        ##### iterating over 3D dictionaries ###############################
        ### pad current input and output images to avoid problems ##########

        current_input = inputul[_]
        current_input = np.expand_dims(current_input, axis =-1)
        current_shape = inputul[_].shape
        current_output = outputul[_]
        ### need to expand the dimension of the output to match the routine        
        current_output = np.expand_dims(current_output, axis =-1)

        if use_masks:
            current_mask = np.ones_like(current_output)
            current_mask = np.array(current_mask, dtype=np.bool)

        ### need to perform the one hot encoding here 
        if dim_output>1:

            current_output = one_hot_encoder(input = np.reshape(current_output,(-1,1)),
                dim_output = dim_output, list_values= [kkt for kkt in range(dim_output)])
            current_output = np.reshape(current_output, (current_shape[0], current_shape[1], current_shape[2], dim_output))

        #current_mask = masks[_]
        #current_mask = np.expand_dims(current_mask, axis =-1)

        #### shape of current image ####
        
        '''
        ### Warning! -- if uncommented, needs to be updated for 3D data ###

        #################################################################################################################
        #### random places being extracted -- most likely not containing any segmentation besides background class ######
        #################################################################################################################

        list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), num_patches_per_subject)
        list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), num_patches_per_subject)

        for __ in range(num_patches_per_subject):
            
            #### iterate over the 2 locations of the 3D cubes #####
            central_points = [list_of_random_places1[__], list_of_random_places2[__]]

            current_input_padded, current_output_padded, current_mask_padded, central_points = check_and_add_zero_padding_2d_image(current_input, 
                current_output, current_mask, central_points, semi_block_size_input, semi_block_size_input2)

            list_blocks_segmentation.append(crop_2D_block(current_output_padded, central_points,
                semi_block_size_output, semi_block_size_output2))
            list_blocks_input.append(crop_2D_block(current_input_padded, central_points,
                semi_block_size_input, semi_block_size_input2))
            list_blocks_masks.append(crop_2D_block(current_mask_padded, central_points,
                semi_block_size_output, semi_block_size_output2))
        '''

        ###############################################################################################
        ###### specifically extract 3D blocks with a non-background class #############################
        ###############################################################################################

        for class_number in range(1,dim_output):

            ####################################
            ##### Class number class_number ####
            ####################################

            ### remainder we already have one hot encoding done above

            indices_class = np.where(current_output[...,class_number] == 1)
            indices_class_dim1 = indices_class[0]
            indices_class_dim2 = indices_class[1]
            indices_class_dim3 = indices_class[2]

            if len(indices_class_dim1)==0:

                print('class '+str(class_number)+' not found in current image')

            else:
                        
                list_of_random_places = random.sample(range(0,len(indices_class_dim1)), num_patches_per_subject)

                for __ in range(num_patches_per_subject):

                    central_points = [indices_class_dim1[list_of_random_places[__]],
                        indices_class_dim2[list_of_random_places[__]], indices_class_dim3[list_of_random_places[__]]]

                    if use_masks:
                        current_input_padded, current_output_padded, current_mask_padded, central_points = check_and_add_zero_padding_3d_image(current_input,
                            current_output, current_mask,central_points, semi_block_size_input, semi_block_size_input2, use_masks)
                    else:
                        current_input_padded, current_output_padded, central_points = check_and_add_zero_padding_3d_image(current_input,
                            current_output, None, central_points, semi_block_size_input, semi_block_size_input2, use_masks)                       
                    for plm_bds in range(len(list_dim_last_layer)):
        
                        ### Hack to ensure that the labels sum to 1 at every voxel ###
                        current_segmentation_block = crop_3D_block(current_output_padded, central_points,
                            dict_semi_block_size_output[plm_bds+1],dict_semi_block_size_output2[plm_bds+1])

                        current_segmentation_block = check_segmentation_block(current_segmentation_block)
                        dict_bds_blocks_segmentation[plm_bds+1].append(current_segmentation_block)
                        if use_masks:
                            dict_bds_blocks_masks[plm_bds+1].append(crop_3D_block(current_mask_padded, central_points,
                                dict_semi_block_size_output[plm_bds+1],dict_semi_block_size_output2[plm_bds+1]))

                    current_segmentation_block = crop_3D_block(current_output_padded, central_points,
                        semi_block_size_output,semi_block_size_output2)
                    current_segmentation_block = check_segmentation_block(current_segmentation_block)

                    list_blocks_segmentation.append(current_segmentation_block)
                    list_blocks_input.append(crop_3D_block(current_input_padded, central_points,
                        semi_block_size_input,semi_block_size_input2))
                    if use_masks:
                        list_blocks_masks.append(crop_3D_block(current_mask_padded, central_points,
                            semi_block_size_output,semi_block_size_output2))

    list_blocks_input = np.stack(list_blocks_input)
    list_blocks_segmentation = np.stack(list_blocks_segmentation)
    if use_masks:
        list_blocks_masks  =np.stack(list_blocks_masks)

    for plm_bds in range(len(list_dim_last_layer)):

        dict_bds_blocks_segmentation[plm_bds+1] = np.stack(dict_bds_blocks_segmentation[plm_bds+1])
        if use_masks:
            dict_bds_blocks_masks[plm_bds+1] = np.stack(dict_bds_blocks_masks[plm_bds+1])

        shape_of_seg = dict_bds_blocks_segmentation[plm_bds+1].shape
        dict_bds_blocks_segmentation[plm_bds+1] = dict_bds_blocks_segmentation[plm_bds+1].reshape((-1,1))
        dict_bds_blocks_segmentation[plm_bds+1] = dict_bds_blocks_segmentation[plm_bds+1].reshape((shape_of_seg[0],
            shape_of_seg[1],shape_of_seg[2],shape_of_seg[3],dim_output))

    shape_of_seg = list_blocks_segmentation.shape
    list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
    #list_blocks_segmentation = output_transformation(list_blocks_segmentation)
    #enc = preprocessing.OneHotEncoder()
    #enc.fit(list_blocks_segmentation)
    #list_blocks_segmentation = enc.transform(list_blocks_segmentation).toarray()
    #list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
    #list_blocks_segmentation = OneHotEncoder(list_blocks_segmentation)
    #list_blocks_segmentation = one_hot_encoder(input = list_blocks_segmentation,dim_output = 2, list_values = [0.0,1.0])

    list_blocks_segmentation = list_blocks_segmentation.reshape((shape_of_seg[0],shape_of_seg[1],shape_of_seg[2],shape_of_seg[3],dim_output))

    output_bds_seg = []
    if use_masks:
        bds_masks = []
    for key in dict_bds_blocks_segmentation.keys():
        output_bds_seg.append(dict_bds_blocks_segmentation[key])
        if use_masks:
            bds_masks.append(dict_bds_blocks_masks[key])

    if use_masks:
       return list_blocks_input, list_blocks_segmentation, list_blocks_masks, output_bds_seg, bds_masks
    else:
       return list_blocks_input, list_blocks_segmentation, output_bds_seg       

def extract_3d_blocks_training_regression(inputul, outputul, iteration, block_size_input, block_size_output, dim_output, num_subjects,
    num_patches_per_subject, masks, list_dim_last_layer, use_masks):

    ## inputul -- shape (num_batch, width, height, num_channels_input)
    ## outputul -- shape (num_batch, width, height, 1)
    ## masks -- shape (num_batch, width, height, 1) ### ones and zeros basically 

    #### this will extract num_subjects * num_patches_per_subject #######

    lista = np.arange(len(inputul.keys()))
    np.random.seed(iteration)
    np.random.shuffle(lista)
    current_index = lista[:num_subjects]

    semi_block_size_input = int(block_size_input//2)
    semi_block_size_input2 = block_size_input - semi_block_size_input
    semi_block_size_output = int(block_size_output//2)
    semi_block_size_output2 = block_size_output - semi_block_size_output
    
    list_blocks_input = []
    list_blocks_segmentation = []
    if use_masks:
        list_blocks_masks = []
    
    ##########################################
    ####### Bayesian Deep Supervision ########
    ##########################################

    dict_bds_blocks_segmentation = defaultdict()
    if use_masks:
        dict_bds_blocks_masks = defaultdict()
    
    dict_semi_block_size_output = defaultdict()
    dict_semi_block_size_output2 = defaultdict()
    
    for _ in range(len(list_dim_last_layer)):
    
        dict_bds_blocks_segmentation[_+1] = []
        if use_masks:
            dict_bds_blocks_masks[_+1] = []      
        dict_semi_block_size_output[_+1] = int(list_dim_last_layer[_]//2)
        dict_semi_block_size_output2[_+1] = list_dim_last_layer[_] - dict_semi_block_size_output[_+1]
        
    for _ in current_index:

        ##### iterating over 3D dictionaries ###############################
        ### pad current input and output images to avoid problems ##########

        current_input = inputul[_]
        current_input = np.expand_dims(current_input, axis =-1)
        current_shape = inputul[_].shape
        current_output = outputul[_]
        current_output = np.ones((current_shape[0], current_shape[1], current_shape[2])) * current_output
        ### need to expand the dimension of the output to match the routine
        current_output = np.expand_dims(current_output, axis =-1)
        ### need to perform the one hot encoding here 
        if dim_output>1:

            current_output = one_hot_encoder(input = np.reshape(current_output,(-1,1)),
                dim_output = dim_output, list_values= [kkt for kkt in range(1,dim_output+1)])
            current_output = np.reshape(current_output, (current_shape[0], current_shape[1], current_shape[2], dim_output))

        if use_masks:
            current_mask = masks[_]
            current_mask = np.expand_dims(current_mask, axis =-1)
        #### shape of current image ####
        
        '''
        ### Warning! -- if uncommented, needs to be updated for 3D data ###

        #################################################################################################################
        #### random places being extracted -- most likely not containing any segmentation besides background class ######
        #################################################################################################################

        list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), num_patches_per_subject)
        list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), num_patches_per_subject)

        for __ in range(num_patches_per_subject):
            
            #### iterate over the 2 locations of the 3D cubes #####
            central_points = [list_of_random_places1[__], list_of_random_places2[__]]

            current_input_padded, current_output_padded, current_mask_padded, central_points = check_and_add_zero_padding_2d_image(current_input, 
                current_output, current_mask, central_points, semi_block_size_input, semi_block_size_input2)

            list_blocks_segmentation.append(crop_2D_block(current_output_padded, central_points,
                semi_block_size_output, semi_block_size_output2))
            list_blocks_input.append(crop_2D_block(current_input_padded, central_points,
                semi_block_size_input, semi_block_size_input2))
            list_blocks_masks.append(crop_2D_block(current_mask_padded, central_points,
                semi_block_size_output, semi_block_size_output2))
        '''

        ###############################################################################################
        ###### specifically extract 3D blocks with a non-background class #############################
        ###############################################################################################



        list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), 
            num_patches_per_subject)
        list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), 
            num_patches_per_subject)
        list_of_random_places3 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), 
            num_patches_per_subject)

        for __ in range(num_patches_per_subject):

            central_points = [list_of_random_places1[__],
                list_of_random_places2[__], list_of_random_places3[__]]
            if use_masks:
                current_input_padded, current_output_padded, current_mask_padded, central_points = check_and_add_zero_padding_3d_image(current_input,
                    current_output, current_mask,central_points, semi_block_size_input, semi_block_size_input2, use_masks)
            else:
                current_input_padded, current_output_padded, central_points = check_and_add_zero_padding_3d_image(current_input,
                    current_output, None, central_points, semi_block_size_input, semi_block_size_input2, use_masks)
            for plm_bds in range(len(list_dim_last_layer)):

                dict_bds_blocks_segmentation[plm_bds+1].append(crop_3D_block(current_output_padded, central_points,
                    dict_semi_block_size_output[plm_bds+1],dict_semi_block_size_output2[plm_bds+1]))
                if use_masks:
                    dict_bds_blocks_masks[plm_bds+1].append(crop_3D_block(current_mask_padded, central_points,
                        dict_semi_block_size_output[plm_bds+1],dict_semi_block_size_output2[plm_bds+1]))

            list_blocks_segmentation.append(crop_3D_block(current_output_padded, central_points,
                semi_block_size_output,semi_block_size_output2))
            list_blocks_input.append(crop_3D_block(current_input_padded, central_points,
                semi_block_size_input,semi_block_size_input2))
            if use_masks:
                list_blocks_masks.append(crop_3D_block(current_mask_padded, central_points,
                    semi_block_size_output,semi_block_size_output2))

    list_blocks_input = np.stack(list_blocks_input)
    list_blocks_segmentation = np.stack(list_blocks_segmentation)
    if use_masks:
        list_blocks_masks  =np.stack(list_blocks_masks)

    for plm_bds in range(len(list_dim_last_layer)):

        dict_bds_blocks_segmentation[plm_bds+1] = np.stack(dict_bds_blocks_segmentation[plm_bds+1])
        if use_masks:
            dict_bds_blocks_masks[plm_bds+1] = np.stack(dict_bds_blocks_masks[plm_bds+1])

        shape_of_seg = dict_bds_blocks_segmentation[plm_bds+1].shape
        dict_bds_blocks_segmentation[plm_bds+1] = dict_bds_blocks_segmentation[plm_bds+1].reshape((-1,1))
        dict_bds_blocks_segmentation[plm_bds+1] = dict_bds_blocks_segmentation[plm_bds+1].reshape((shape_of_seg[0],
            shape_of_seg[1],shape_of_seg[2],shape_of_seg[3],dim_output))

    shape_of_seg = list_blocks_segmentation.shape
    list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
    #list_blocks_segmentation = output_transformation(list_blocks_segmentation)
    #enc = preprocessing.OneHotEncoder()
    #enc.fit(list_blocks_segmentation)
    #list_blocks_segmentation = enc.transform(list_blocks_segmentation).toarray()
    #list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
    #list_blocks_segmentation = OneHotEncoder(list_blocks_segmentation)
    #list_blocks_segmentation = one_hot_encoder(input = list_blocks_segmentation,dim_output = 2, list_values = [0.0,1.0])

    list_blocks_segmentation = list_blocks_segmentation.reshape((shape_of_seg[0],shape_of_seg[1],shape_of_seg[2],shape_of_seg[3],dim_output))

    output_bds_seg = []
    if use_masks:
        bds_masks = []
    for key in dict_bds_blocks_segmentation.keys():
        output_bds_seg.append(dict_bds_blocks_segmentation[key])
        if use_masks:
            bds_masks.append(dict_bds_blocks_masks[key])

    if use_masks:
        return list_blocks_input, list_blocks_segmentation, list_blocks_masks, output_bds_seg, bds_masks
    else:
        return list_blocks_input, list_blocks_segmentation, output_bds_seg 

##################################################################
##################################################################
################# Testing time patch extraction ##################
##################################################################
##################################################################

######  Extract non-overlapping 3D patches in segmentation space #############
###### also extracts the overlapping bigger 3D patches in raw input space ####



def extract_3D_cubes_input_seg(input_image, output_image, semi_block_size_input1, semi_block_size_output1,
    semi_block_size_input2, semi_block_size_output2, dim_output, mask, use_masks):

    input_image = np.expand_dims(input_image, axis =-1)
    current_shape = input_image.shape
    output_image = np.expand_dims(output_image, axis =-1)
    ### need to perform the one hot encoding here 
    
    if dim_output>1:

        output_image = one_hot_encoder(input = np.reshape(output_image, (-1,1)),
            dim_output = dim_output, list_values= [kkt for kkt in range(dim_output)])
        output_image = np.reshape(output_image, (current_shape[0], current_shape[1], current_shape[2], dim_output))

    if use_masks:
        mask = np.expand_dims(mask, axis =-1)

    block_size_output = semi_block_size_output1 + semi_block_size_output2
    block_size_input = semi_block_size_input1 + semi_block_size_input2
    diff_semi_block = np.abs(semi_block_size_output1 - semi_block_size_input1)

    shape_of_data = output_image.shape

    ### dimension 1 ###
    diff_dim1 = shape_of_data[0] %  block_size_output
    if diff_dim1!=0:
        diff_dim1 = block_size_output - diff_dim1

    ### dimension 2 ###
    diff_dim2 = shape_of_data[1] % block_size_output
    if diff_dim2!=0:
        diff_dim2 = block_size_output - diff_dim2

    ### dimension 3 ###
    diff_dim3 = shape_of_data[2] % block_size_output
    if diff_dim3!=0:
        diff_dim3 = block_size_output - diff_dim3


    #####################################################################
    ### pad output space so that it is divisible by block_size_output ###
    #####################################################################

    output_image = np.pad(array = output_image, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant')
    if use_masks:
        mask = np.pad(array = mask, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant')
    input_image = np.pad(array = input_image, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant', constant_values = -1.0)

    ###################################################################################################################################
    ###### Remainder -- to get from output_image_coordinates to input_image_coordinates add on all dimensions + diff_semi_block #######
    ###################################################################################################################################

    ######  Extract non-overlapping 3D cubes in Regression output space ###############################################
    ###### also extracts the overlapping bigger 3D cubes in raw input space ###########################################
    ###### extracts non-overlapping 3D blocks contrained by an ROI box  ###############################################

    shape_of_input_data = input_image.shape
    shape_of_output_data = output_image.shape


    num_cubes_dim1 = np.int(shape_of_output_data[0] // block_size_output)
    num_cubes_dim2 = np.int(shape_of_output_data[1] // block_size_output)
    num_cubes_dim3 = np.int(shape_of_output_data[2] // block_size_output)    

    list_input_cubes = []
    list_output_cubes = []

    for i in range(num_cubes_dim1):
        for j in range(num_cubes_dim2):
            for k in range(num_cubes_dim3):
                
                ##################################################
                ### extract segmentation output space 3D block ###
                ##################################################

                list_output_cubes.append(output_image[block_size_output*i:(block_size_output*(i+1)),
                    block_size_output*j:(block_size_output*(j+1)), block_size_output*k:(block_size_output*(k+1)),:]  )
            
                ########################################
                ### extract raw input space 2D patch ###
                ########################################

                #### it might require some additional zero-padding if near the margins of the image #####

                list_input_cubes.append(check_and_add_zero_padding_3d_image_test_time(input_image = input_image, output_image = output_image,
                    lower_points = [block_size_output*i, block_size_output*j, block_size_output*k],
                    upper_points = [block_size_output*(i+1), block_size_output*(j+1), block_size_output*(k+1)],
                    diff_semi_block_size1 = diff_semi_block, diff_semi_block_size2 = diff_semi_block))
        

    list_output_cubes = np.stack(list_output_cubes)
    list_input_cubes = np.stack(list_input_cubes)

    if use_masks:
        return list_input_cubes, list_output_cubes, shape_of_output_data, mask, output_image, input_image
    else:
        return list_input_cubes, list_output_cubes, shape_of_output_data, output_image, input_image

def check_and_add_zero_padding_3d_image_test_time(input_image, output_image,
    lower_points, upper_points, diff_semi_block_size1, diff_semi_block_size2):

    #### checks if extracting a patch need padding or not 
    #### accounts for the case where the central_points are close to the boundary of the image and expands it with the minimum of the image

    ### image -- shape (height, width, depth, channels)
    ### lower_points -- (c1, c2, c3)
    ### upper_points -- (c1, c2, c3)
    ### semi_block_size -- (l1, l2, l3)

    current_shape = input_image.shape 
    #min_value_image = np.min(input_image)
    min_value_image = -1.0
    padding_dimensions = []
    control=0				

    for _ in range(3):

        dim_list = []

        if lower_points[_] - diff_semi_block_size1 < 0:			
            dim_list.append(np.abs(lower_points[_]-diff_semi_block_size1))
            control+=1

        else:

            dim_list.append(0)

        if upper_points[_] + diff_semi_block_size2 > current_shape[_]:
            dim_list.append(np.abs(upper_points[_]+diff_semi_block_size2 - current_shape[_]))
            control+=1
        else:
            dim_list.append(0)

        padding_dimensions.append(tuple(dim_list))

    if control > 0:

        padding_dimensions = tuple(padding_dimensions)
        padding_dimensions_extra = list(padding_dimensions)
        padding_dimensions_extra.append(tuple([0,0]))
        padding_dimensions_extra = tuple(padding_dimensions_extra)
  
        ###################################
        ##### crop as much as possible ####
        ###################################
        
        c1 = np.abs(padding_dimensions_extra[0][0]-diff_semi_block_size1)
        c2 = np.abs(padding_dimensions_extra[0][1]-diff_semi_block_size1)
        c3 = np.abs(padding_dimensions_extra[1][0]-diff_semi_block_size1)
        c4 = np.abs(padding_dimensions_extra[1][1]-diff_semi_block_size1)
        c5 = np.abs(padding_dimensions_extra[2][0]-diff_semi_block_size1)
        c6 = np.abs(padding_dimensions_extra[2][1]-diff_semi_block_size1)        
   

        patch = input_image[lower_points[0]- c1: upper_points[0]+c2,
            lower_points[1]-c3 : upper_points[1]+c4, lower_points[2]-c5 : upper_points[2]+c6 ,:]

        #######################################
        ##### afterwards pad the remainder ####
        #######################################

        patch = np.pad(patch, padding_dimensions_extra, mode='constant',
            constant_values = min_value_image)

    else:

        patch = input_image[lower_points[0]-diff_semi_block_size1 : upper_points[0]+diff_semi_block_size1,
            lower_points[1]-diff_semi_block_size2 : upper_points[1]+diff_semi_block_size2,
            lower_points[2]-diff_semi_block_size2 : upper_points[2]+diff_semi_block_size2, :]

    return patch




def extract_3D_cubes_input_seg_regression(input_image, semi_block_size_input1, semi_block_size_output1,
    semi_block_size_input2, semi_block_size_output2, dim_output, mask, use_masks):

    input_image = np.expand_dims(input_image, axis =-1)
    current_shape = input_image.shape
    ### need to perform the one hot encoding here 
    if use_masks:
        mask = np.expand_dims(mask, axis =-1)

    block_size_output = semi_block_size_output1 + semi_block_size_output2
    block_size_input = semi_block_size_input1 + semi_block_size_input2
    diff_semi_block = np.abs(semi_block_size_output1 - semi_block_size_input1)

    shape_of_data = input_image.shape

    ### dimension 1 ###
    diff_dim1 = shape_of_data[0] %  block_size_output
    if diff_dim1!=0:
        diff_dim1 = block_size_output - diff_dim1
 
    ### dimension 2 ###
    diff_dim2 = shape_of_data[1] % block_size_output
    if diff_dim2!=0:
        diff_dim2 = block_size_output - diff_dim2

    ### dimension 3 ###
    diff_dim3 = shape_of_data[2] % block_size_output
    if diff_dim3!=0:
        diff_dim3 = block_size_output - diff_dim3
 

    #####################################################################
    ### pad output space so that it is divisible by block_size_output ###
    #####################################################################

    if use_masks:
        mask = np.pad(array = mask, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant')
    input_image = np.pad(array = input_image, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant')

    ###################################################################################################################################
    ###### Remainder -- to get from output_image_coordinates to input_image_coordinates add on all dimensions + diff_semi_block #######
    ###################################################################################################################################

    ######  Extract non-overlapping 3D cubes in Regression output space ###############################################
    ###### also extracts the overlapping bigger 3D cubes in raw input space ###########################################
    ###### extracts non-overlapping 3D blocks contrained by an ROI box  ###############################################

    shape_of_input_data = input_image.shape

    
    num_cubes_dim1 = np.int(shape_of_input_data[0] // block_size_output)
    num_cubes_dim2 = np.int(shape_of_input_data[1] // block_size_output)
    num_cubes_dim3 = np.int(shape_of_input_data[2] // block_size_output)    

    list_input_cubes = []
    list_output_cubes = []

    for i in range(num_cubes_dim1):
        for j in range(num_cubes_dim2):
            for k in range(num_cubes_dim3):
        

                ########################################
                ### extract raw input space 2D patch ###
                ########################################

                #### it might require some additional zero-padding if near the margins of the image #####

                list_input_cubes.append(check_and_add_zero_padding_3d_image_test_time_regression(input_image = input_image, 
                    lower_points = [block_size_output*i, block_size_output*j, block_size_output*k],
                    upper_points = [block_size_output*(i+1), block_size_output*(j+1), block_size_output*(k+1)],
                    diff_semi_block_size1 = diff_semi_block, diff_semi_block_size2 = diff_semi_block))
        
    list_input_cubes = np.stack(list_input_cubes)

    if use_masks:
        return list_input_cubes, shape_of_input_data, mask, input_image
    else:
        return list_input_cubes, shape_of_input_data, input_image

def check_and_add_zero_padding_3d_image_test_time_regression(input_image,
    lower_points, upper_points, diff_semi_block_size1, diff_semi_block_size2):

    #### checks if extracting a patch need padding or not 
    #### accounts for the case where the central_points are close to the boundary of the image and expands it with the minimum of the image

    ### image -- shape (height, width, depth, channels)
    ### lower_points -- (c1, c2, c3)
    ### upper_points -- (c1, c2, c3)
    ### semi_block_size -- (l1, l2, l3)

    current_shape = input_image.shape 
    min_value_image = np.min(input_image)
    padding_dimensions = []
    control=0				

    for _ in range(3):

        dim_list = []

        if lower_points[_] - diff_semi_block_size1 < 0:			
            dim_list.append(np.abs(lower_points[_]-diff_semi_block_size1))
            control+=1

        else:

            dim_list.append(0)

        if upper_points[_] + diff_semi_block_size2 > current_shape[_]:
            dim_list.append(np.abs(upper_points[_]+diff_semi_block_size2 - current_shape[_]))
            control+=1
        else:
            dim_list.append(0)

        padding_dimensions.append(tuple(dim_list))

    if control > 0:

        padding_dimensions = tuple(padding_dimensions)
        padding_dimensions_extra = list(padding_dimensions)
        padding_dimensions_extra.append(tuple([0,0]))
        padding_dimensions_extra = tuple(padding_dimensions_extra)
   
        ###################################
        ##### crop as much as possible ####
        ###################################
        
        c1 = np.abs(padding_dimensions_extra[0][0]-diff_semi_block_size1)
        c2 = np.abs(padding_dimensions_extra[0][1]-diff_semi_block_size1)
        c3 = np.abs(padding_dimensions_extra[1][0]-diff_semi_block_size1)
        c4 = np.abs(padding_dimensions_extra[1][1]-diff_semi_block_size1)
        c5 = np.abs(padding_dimensions_extra[2][0]-diff_semi_block_size1)
        c6 = np.abs(padding_dimensions_extra[2][1]-diff_semi_block_size1)        
    

        patch = input_image[lower_points[0]- c1: upper_points[0]+c2,
            lower_points[1]-c3 : upper_points[1]+c4, lower_points[2]-c5 : upper_points[2]+c6 ,:]

        #######################################
        ##### afterwards pad the remainder ####
        #######################################

        patch = np.pad(patch, padding_dimensions_extra, mode='constant',
            constant_values = min_value_image)
 
    else:

        patch = input_image[lower_points[0]-diff_semi_block_size1 : upper_points[0]+diff_semi_block_size1,
            lower_points[1]-diff_semi_block_size2 : upper_points[1]+diff_semi_block_size2,
            lower_points[2]-diff_semi_block_size2 : upper_points[2]+diff_semi_block_size2, :]

    return patch



