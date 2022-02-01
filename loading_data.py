import numpy as np
from collections import defaultdict
import nibabel as nib
import pandas as pd


def parse_string_list(string_list, index):

	new_list = [string_list[index_now] for index_now in index]

	return new_list


def parse_info(list_of_nifty_files_gm, subject_info, list_extract_subjects):

	subject_info.set_index("Subject", inplace=True)

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################
	
	lista_outcomes = defaultdict()
	
	
	control=0
	for current_subject in list_extract_subjects:

		print(current_subject)
		#try:
	
		############################################
		###### check if there is a nifty file ######
		############################################
		plm = [s for s in list_of_nifty_files_gm if current_subject in s]

		if len(plm)>0:

			current_row_of_interest = subject_info.loc[current_subject]
			lista_outcomes[control] = current_row_of_interest.Age
			control+=1
			
		else:
			print('Did not have nifty file ')
		#except:
		
		#	print('We could not find the nifti files')

	return lista_outcomes



def data_factory_whole_brain(list_of_nifty_files):

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################

	lista_imagini = defaultdict()

	control = 0
	for nifty_file in list_of_nifty_files:

		print('loading ... '+str(nifty_file))
		temporar_object = nib.load(nifty_file)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()

		lista_imagini[control] = np.squeeze(temporar_data)
		print(lista_imagini[control].shape)
		control+=1

	return lista_imagini


def data_factory_slice_brain(list_of_nifty_files, num_slice, dim):

	lista_imagini = []

	for nifty_file in list_of_nifty_files:

		print('loading ... '+str(nifty_file))
		temporar_object = nib.load(nifty_file)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()

		if dim==0:
				
			lista_imagini.append(np.squeeze(temporar_data[num_slice,:,:]))

		elif dim==1:
				
			lista_imagini.append(np.squeeze(temporar_data[:,num_slice,:]))

		elif dim==2:
				
			### Warning -- this contains a hack for the IBRS dataset	
			lista_imagini.append(np.squeeze(temporar_data[:,:,num_slice]))

		print(lista_imagini[-1].shape)
	lista_imagini = np.stack(lista_imagini)

	return lista_imagini

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

def whitening(image):

    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret

def data_factory_whole_brain_training(list_of_nifty_files,
	subject_info, list_extract_subjects):

	subject_info.set_index("Subject", inplace=True)

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################

	lista_imagini = defaultdict()
	lista_outcomes = defaultdict()
	lista_name = defaultdict()

	### parse the nifty lists for the ones present in list_extract_subjects ###
	list_parsed = []

	control=0
	for current_subject in list_extract_subjects:

		print(current_subject)
		#try:
		############################################
		###### check if there is a nifty file ######
		############################################
		plm = [s for s in list_of_nifty_files if current_subject in s]

		if len(plm)>0:

			list_parsed.append( [s for s in list_of_nifty_files if current_subject in s][0])			
			current_row_of_interest = subject_info.loc[current_subject]
			lista_outcomes[control] = current_row_of_interest.Age
			print(current_row_of_interest.Age)
			lista_name[control] = current_subject
			control+=1
			
		else:
			print('Did not have nifty file ')
		

	##### load data #####
	control = 0
	for nifty_file in list_parsed:
		print('name of subject')
		print(lista_name[control])
		print('loading ... '+str(nifty_file))

		temporar_object = nib.load(nifty_file)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()

		lista_imagini[control] = temporar_data
		control+=1

	return lista_imagini, lista_outcomes, list_parsed, lista_name


