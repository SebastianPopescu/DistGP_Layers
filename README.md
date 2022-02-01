>📋  Accompanying code for "A U-Net model for Local Brain Age"

# A U-Net model for Local Brain Age

This repository is the official implementation of [Distributional Gaussian Process Layers for Out-of-Distribution Detection](https://link.springer.com/chapter/10.1007/978-3-030-78191-0_32). 

>📋  ![Schematic of DistGP Layer](schematic_dist_gp_layer.png)

>📋  ![Segmentation Architecture](segmentation_network.png)

## Requirements

To install requirements:

```setup
pip install tensorflow-gpu
```
## Training

To train the model in the paper, run this command:

```train
python3 full_training_script.py --num_encoding_layers=2 --num_filters=64 --num_subjects=2 --num_voxels_per_subject=2 --location_metadata=$absolute_path_of_metadata_for_dataset --dirpath_gm=$absolute_path_of_directory_of_spm12_processed_gray_matter_nifti_files
--dirpath_wm=$absolute_path_of_directory_of_spm12_processed_white_matter_nifti_files --dataset_name=$dataset_name
```

>📋  The above command line function start training the U-Net model provided a dataset that already went through spm12's Dartel pipeline. One can get the folder with gm and wm nifti files by using  batched_spm12_dartel(img_dir, name_of_dataset, size_batch) function inside dartel_pipeline.py. We mention that the training process is quite lengthy (around 3 weeks on a GPU) if one wants the best possible performance.

## Evaluation

To evaluate our already trained U-Net model on your dataset you need run the following command (if python throws error such as too many files opened, just type ulimit -n 10000 on the command line)

```eval
ython3 full_testing_script.py --num_encoding_layers=2 --num_filters=64 --num_subjects=2 --num_voxels_per_subject=2 --location_metadata=$absolute_path_of_metadata_for_dataset --dirpath_gm=$absolute_path_of_directory_of_spm12_processed_gray_matter_nifti_files
--dirpath_wm=$absolute_path_of_directory_of_spm12_processed_white_matter_nifti_files --dataset_name=$dataset_name
```

>📋As mentioned in the paper, the brain scans have to go through the Dartel pipeline in spm12. <br/>
>in "spm_brain_age_preprocess_b23d.m" you need to change "/data/my_programs/spm12" path to suit the location of where your local spm12 is installed. <br/>
>In LocalBrainAge_testing.py the format of the .csv file containing meta-data has to have the following column names "Age", "Gender", "Subject", alternatively modify the code at lines 62-64. Important: the "Gender" column has to be coded with 1 for males, respectively 0 for females. <br/>
>in your local spm12, you need to copy the files situated in the templates folder (Template_{1,2,3,4,5,6}.nii) to "$your_spm12_folder/templates/". Warning! On some workstations calling the above will only result in the spm12 preprocessing part going through and stopping after (without actually running the LocalBrainAge part). If this happens, go in full_testing_script.py and comment out line 20 ( #batched_spm12_dartel(img_dir = args.dirpath_raw_data, name_of_dataset = args.dataset_name, size_batch = args.size_batch_preprocessing)) and then run again the above command line.

## Pre-trained Models

The pre-trained model can be downloaded [here](https://drive.google.com/drive/folders/1htVlUTyWI2fN6Hz0inBJevlYV0zruOc2?usp=sharing) and should be placedinside the saved_model_3D_UNET_Dropout/iteration_870000/ folder

## Citing this work

>📋 If you use these tools or datasets in your publications, please consider citing the accompanying paper with a BibTeX entry similar to the following:

```
@InProceedings{10.1007/978-3-030-78191-0_32,
author="Popescu, Sebastian G.
and Sharp, David J.
and Cole, James H.
and Kamnitsas, Konstantinos
and Glocker, Ben",
editor="Feragen, Aasa
and Sommer, Stefan
and Schnabel, Julia
and Nielsen, Mads",
title="Distributional Gaussian Process Layers for Outlier Detection in Image Segmentation",
booktitle="Information Processing in Medical Imaging",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="415--427",
abstract="We propose a parameter efficient Bayesian layer for hierarchical convolutional Gaussian Processes that incorporates Gaussian Processes operating in Wasserstein-2 space to reliably propagate uncertainty. This directly replaces convolving Gaussian Processes with a distance-preserving affine operator on distributions. Our experiments on brain tissue-segmentation show that the resulting architecture approaches the performance of well-established deterministic segmentation algorithms (U-Net), which has never been achieved with previous hierarchical Gaussian Processes. Moreover, by applying the same segmentation model to out-of-distribution data (i.e., images with pathology such as brain tumors), we show that our uncertainty estimates result in out-of-distribution detection that outperforms the capabilities of previous Bayesian networks and reconstruction-based approaches that learn normative distributions.",
isbn="978-3-030-78191-0"
}



```







