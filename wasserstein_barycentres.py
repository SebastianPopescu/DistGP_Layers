import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

##########################################################################################################
###### Iterative Operation as described in A fixed-point approach to barycenters in Wasserstein space ####
##########################################################################################################

def wasserstein_barycentre_gaussian_measures(gaussian_means, gaussian_vars, barycentric_coordinates, num_iterations):

    ### gaussian_means -- [num_batches ,num_measures, dim_input]
    ### gaussian_vars -- [num_batches ,num_measures, dim_input, dim_input]
    ### barycentric_coordinates -- [num_batches, num_measures, 1] 
    ### num_iterations -- scalar

    #### Simple case when dim_input=1 and algorithm converges in one iteration ##
    shape_of_gaussian_means = gaussian_means.get_shape().as_list()
    num_measures = shape_of_gaussian_means[1]
    dim_input = shape_of_gaussian_means[2]

    tiled_barycentric_coordinates = tf.tile(barycentric_coordinates, [1, 1, dim_input])
    ### tiled_barycentric_coordinates -- [num_batches, num_measures, dim_input]    

    if dim_input==1:

        barycentre_mean = tf.multiply(gaussian_means, tiled_barycentric_coordinates)
        barycentre_mean = tf.squeeze(barycentre_mean, axis = -1)
        ### barycentre_mean -- [num_batches, num_measures]    
        barycentre_mean = tf.reduce_sum(barycentre_mean, axis = -1 , keepdims = True)
        ### barycentre_mean -- [num_batches, 1]  


        gaussian_vars = tf.squeeze(gaussian_vars, axis =-1)
        #gaussian_vars -- [num_batches, num_measures, dim_input]
        barycentre_var = tf.multiply(barycentric_coordinates,tf.sqrt(gaussian_vars))
        #barycentre_var -- [num_batches, num_measures, dim_input]       
        barycentre_var = tf.square(tf.reduce_sum(barycentre_var, axis = 1, keepdims=False))
        #barycentre_var -- [num_batches, dim_input] 

    else:

        #### TODO -- we still need to update this 

        barycentre_mean = tf.multiply(gaussian_means, barycentric_coordinates)
        barycentre_mean = tf.reduce_sum(barycentre_mean)

        ### Iterative opeartion starting with S0 identity matrix
        barycentre_var = tf.ones(dim_input)
        barycentric_coordinates = tf.expand_dims(barycentric_coordinates, axis =-1)

        for _ in range(num_iterations):

            sq_root_barycentre_var = tf.linalg.sqrtm(barycentre_var)
            inverse_sq_root_barycentre_var = tf.inv(sq_root_barycentre_var)
            tiled_sq_root_barycentre_var = tf.tile(tf.expand_dims(sq_root_barycentre_var, axis = 0),
                [num_measures,1,1])

            inner_operations = tf.matmul(tf.matmul(tiled_sq_root_barycentre_var, gaussian_vars),tiled_sq_root_barycentre_var)
            inner_operations = tf.linalg.sqrtm(inner_operations)
            inner_operations = barycentric_coordinates * inner_operations
            inner_operations = tf.reduce_sum(inner_operations,axis=0,keepdims=False)
            inner_operations = tf.matmul(inner_operations, inner_operations)

            barycentre_var = tf.matmul(tf.matmul(inverse_sq_root_barycentre_var,inner_operations),inverse_sq_root_barycentre_var)

    #barycentre_mean -- [num_batches, dim_input] 
    #barycentre_var -- [num_batches, dim_input] 
    
    return barycentre_mean, barycentre_var
