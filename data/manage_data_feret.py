import h5py
import numpy as np
import os
import tqdm
import pandas as pd


def get_embeddings(embeddings_path):
	# return embeddings and the list of filenames, in the same df
	embeddings = []
	files = os.listdir(embeddings_path)

	for file in tqdm.tqdm(files):
		# store embeddings normalized with L2-norm
		embedding = np.load(os.path.join(embeddings_path, file))
		embedding = embedding / np.linalg.norm(embedding, ord=2)
		embeddings.append(embedding)

	# remove file extension (npy)
	files = [f[:-4] for f in files]
	# create DataFrame with filenames and embeddings, maintaining the order of files
	embeddings = np.array(embeddings)
	files_embeddings_df = pd.DataFrame(embeddings, columns=['f'+str(i) for i in range(len(embedding))])
	files_embeddings_df['filename'] = files

	return files_embeddings_df, len(embedding)


def code_age(age):
	# to generate three classes for age
	return [min(int((int(int(a)/10) - 1)/2), 2) for a in age]


def code_sex(sex):
	return [(0 if s == 'Male' else 1) for s in sex]


def code_eth(eth):
	new_eth = []
	for e in eth:
		if e in ['Asian']:
			new_eth.append(0)
		elif e in ['Black-or-African-American']:
			new_eth.append(1)
		elif e in ['White']:
			new_eth.append(2)
		else:
			new_eth.append(3)
	return new_eth


def manage_annotation_file(annotation_file):
	df = pd.read_csv(annotation_file)
	# remove files not used in the experiments
	df = df.drop(['lec_x', 'lec_y', 'rec_x', 'rec_y'], axis=1)
	df['filename'] = [f[:-4] for f in list(df['filename'])]

	# code the labels of interest
	df['age'] = code_age(list(df['age']))
	df['sex'] = code_sex(list(df['sex']))
	df['ethnicity'] = code_eth(list(df['ethnicity']))
	return df


def get_ordered_labels(ord_df, annotation, length_embeddings):
	ord_lab_df = ord_df.merge(annotation, on='filename', how='left')
	return ord_lab_df[['filename', 'sex', 'age', 'ethnicity'] + ['f' + str(i) for i in range(length_embeddings)]]


def get_feret_df(embeddings_path, seed, save_files=False):
	ordered_filenames_embeddings_df, length_embeddings = get_embeddings(embeddings_path)
	annotation = manage_annotation_file('data/annotation.csv')
	ordered_filenames_lab_df = get_ordered_labels(ordered_filenames_embeddings_df, annotation, length_embeddings)

	ordered_filenames_lab_df = ordered_filenames_lab_df.sample(frac=1, random_state=seed).reset_index(drop=True)

	if save_files:
		ordered_filenames_lab_df.to_csv('data/feret_df.csv', index=False)

	return ordered_filenames_lab_df


def get_x_ready(df, length_embeddings):
	# get the embeddings
	x = df[['f'+str(i) for i in range(length_embeddings)]]
	x = x.to_numpy()
	return x


def get_y_ready(df, labels):
	# get the labels
	y = df[labels]
	y = y.to_numpy()
	return y


