import h5py
import numpy as np
import os
import tqdm
import pandas as pd

def get_embeddings(embeddings_path):
	# return embeddings and the list of filenames, in the same order
	embeddings = []
	files = os.listdir(embeddings_path)

	for file in tqdm.tqdm(files):
		with h5py.File(os.path.join(embeddings_path, file), "r") as f:

			# get first object name/key; may or may NOT be a group
			a_group_key = list(f.keys())[0]

			# This gets the object names in the group and returns as a list
			data = list(f[a_group_key])

			embeddings.append(data)

	# remove file extension (h5)
	files = [f[:-3] for f in files]
	# create DataFrame with filenames and embeddings, maintaining the order of files
	files_embeddings_df = pd.DataFrame(zip(files, embeddings), columns=['filename', 'embeddings'])

	return files_embeddings_df


def code_age(age):
	return [(int(int(a)/10) - 1) for a in age]


def code_sex(sex):
	return [(0 if s == 'Male' else 1) for s in sex]


def code_eth(eth):
	new_eth = []
	for e in eth:
		if e == 'White':
			new_eth.append(0)
		elif e in ['Asian', 'Asian-Middle-Eastern', 'Pacific-Islander', 'Asian-Southern']:
			new_eth.append(1)
		elif e == 'Black-or-African-American':
			new_eth.append(2)
		elif e == 'Hispanic':
			new_eth.append(3)
		else:
			new_eth.append(4)
	return new_eth


def manage_annotation_file(annotation_file):
	df = pd.read_csv(annotation_file)
	# remove files not used in the experiments
	df = df.dropna()

	# code the labels of interest
	df['age'] = code_age(list(df['age']))
	df['sex'] = code_sex(list(df['sex']))
	df['ethnicity'] = code_eth(list(df['ethnicity']))
	return df


def get_ordered_labels(ord_df, annotation):
	ord_lab_df = ord_df.merge(annotation, on='filename', how='left')
	return ord_lab_df[['filename', 'embeddings', 'sex', 'age', 'ethnicity']]


def get_train_test_df(ord_lab_df, pct=0.7):
	import random
	random.seed(2)

	# get the list of patients (with repetitions)
	filename_patients = [p[:5] for p in list(ord_lab_df['filename'])]
	# get the list of unique patients and shuffle it
	patients = list(set(filename_patients))
	random.shuffle(patients)

	# divide patients between training and test
	train_pats = patients[:int(len(patients) * pct)]
	test_pats = patients[int(len(patients) * pct):]

	# create a mask to divide ord_lab_df
	train_mask = [(f in train_pats) for f in filename_patients]
	train_df = ord_lab_df[train_mask]
	not_train_mask = [not t for t in train_mask]
	test_df = ord_lab_df[not_train_mask]

	return train_df, test_df


ordered_filenames_embeddings_df = get_embeddings('feret_embeddings')
annotation = manage_annotation_file('annotation.csv')
ordered_filenames_lab_df = get_ordered_labels(ordered_filenames_embeddings_df, annotation)
train_df, test_df = get_train_test_df(ordered_filenames_lab_df)