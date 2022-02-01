import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import defaultdict

def dice_score(predicted_labels, labels, dim_output, type_unet):

    ####### Dice score for binary classification #######

    ### predicted_labels -- shape (num_batch, height, width)
    ### labels -- shape (num_batch, height, width)
    print('shape of predicted labels')
    print(predicted_labels)
    print('shape of actual labels')
    print(labels)

    indices_predictions = tf.round(predicted_labels)
    indices_labels = tf.round(labels)

    print('after transformation')
    print(indices_predictions)
    print(indices_labels)

    dice_score = defaultdict()
    for _ in range(2):
        shared_bool = tf.logical_and( tf.equal(tf.cast(indices_predictions,tf.float32),
            tf.ones_like(indices_predictions, dtype=tf.float32)* tf.cast(_,tf.float32)) ,
            tf.equal(tf.cast(indices_labels,tf.float32),
                tf.ones_like(indices_predictions, dtype=tf.float32)*tf.cast(_,tf.float32)))
        area_shared = tf.reduce_sum(tf.cast(shared_bool,tf.float32))

        predictions_bool = tf.equal(tf.cast(indices_predictions,tf.float32),
            tf.ones_like(indices_predictions, dtype=tf.float32)* tf.cast(_,tf.float32))
        area_predictions = tf.reduce_sum(tf.cast(predictions_bool,tf.float32))
        
        labels_bool = tf.equal(tf.cast(indices_labels,tf.float32),
            tf.ones_like(indices_predictions, dtype=tf.float32)* tf.cast(_,tf.float32))
        area_labels = tf.reduce_sum(tf.cast(labels_bool,tf.float32))

        dice_score[_] = (2.0 * area_shared+1e-6) / (area_predictions + area_labels + 1e-6)

    return dice_score



def dice_score_multiclass(predicted_labels, labels, num_classes, type_unet):

    #### Dice Score for at least 3 classes #####

    ### predicted_labels -- shape (num_batch, height, width, depth, num_classes) if 3D data
    ### labels -- shape (num_batch, height, width, depth, num_classes) if 3D data

    print('shape of predicted labels')
    print(predicted_labels)
    print('shape of actual labels')
    print(labels)

    shape_of_data = labels.get_shape().as_list()
    if type_unet=='3D':

        indices_predictions = tf.argmax(predicted_labels, axis=-1)
        #indices_predictions = tf.reshapes(indices_predictions,[-1 , shape_of_data[1] * shape_of_data[2] * shape_of_data[3] * 1])		

        indices_labels = tf.argmax(labels, axis=-1)
        #indices_labels = tf.reshape(indices_labels,[-1 , shape_of_data[1] * shape_of_data[2] * shape_of_data[3] * 1])		
    
    else:

        indices_predictions = tf.argmax(predicted_labels, axis=-1)
        #indices_predictions = tf.reshape(indices_predictions,[-1 , shape_of_data[1] * shape_of_data[2] * 1])		

        indices_labels = tf.argmax(labels, axis=-1)
        #indices_labels = tf.reshape(indices_labels,[-1 , shape_of_data[1] * shape_of_data[2]  * 1])		

    print('after transformation')
    print(indices_predictions)
    print(indices_labels)

    dice_score = defaultdict()
    for _ in range(num_classes):

        shared_bool = tf.logical_and( tf.equal(tf.cast(indices_predictions,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32)) ,
            tf.equal(tf.cast(indices_labels,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)*tf.cast(_,tf.float32)))
        area_shared = tf.reduce_sum(tf.cast(shared_bool,tf.float32))

        predictions_bool = tf.equal(tf.cast(indices_predictions,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32))
        area_predictions = tf.reduce_sum(tf.cast(predictions_bool,tf.float32))
        
        labels_bool = tf.equal(tf.cast(indices_labels,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32))
        area_labels = tf.reduce_sum(tf.cast(labels_bool,tf.float32))

        dice_score[_] = tf.reduce_mean( (2.0 * area_shared + 1e-6) / (area_predictions + area_labels + 1e-6))

    return dice_score



