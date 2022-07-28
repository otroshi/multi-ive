import tqdm
import os
import numpy as np
import pandas as pd
import random


def _get_label_coded(name):
	age = name.split('_')[0]
	if int(age) < 30:
		return 0
	elif int(age) < 50:
		return 1
	elif int(age) < 80:
		return 2
	else:
		return -1


def get_embeddings(embeddings_path):
	embeddings = []
	labels = []
	# return embeddings and the list of filenames, in the same df
	for path, subdirs, files in tqdm.tqdm(os.walk(embeddings_path)):
		for name in [f for f in files if f.find('.npy') >= 0]:
			# store embeddings normalized with L2-norm
			embedding = np.load(os.path.join(path, name))
			embedding = embedding / np.linalg.norm(embedding, ord=2)
			embeddings.append(embedding)
			# get the labels (contained in the name of the second last folder)
			label = _get_label_coded(name)
			labels.append(label)

	# create DataFrame with filenames, embeddings, users, and files
	embeddings = np.array(embeddings)
	files_embeddings_df = pd.DataFrame(embeddings, columns=['f'+str(i) for i in range(len(embedding))])

	labels = np.array(labels)
	labels_df = pd.DataFrame(labels, columns=['age'])
	files_embeddings_df = pd.concat([files_embeddings_df, labels_df], axis=1)

	return files_embeddings_df, len(embedding)


def get_utkface_df(embeddings_path, seed, save_files=False, limit_size=False):
	ordered_filenames_lab_df, length_embeddings = get_embeddings(embeddings_path)
	ordered_filenames_lab_df['initialIndex'] = ordered_filenames_lab_df.index.values
	if limit_size:
		random.seed(seed)
		indexes_to_keep = []
		# specific for utkface
		for age_code in range(3):
			tmp = ordered_filenames_lab_df.loc[ordered_filenames_lab_df['age'] == age_code]
			indexes = list(tmp['initialIndex'])
			random.shuffle(indexes)
			indexes_to_keep += indexes[:2000]

		ordered_filenames_lab_df = ordered_filenames_lab_df[ordered_filenames_lab_df['initialIndex'].isin(indexes_to_keep)]

	ordered_filenames_lab_df = ordered_filenames_lab_df.sample(frac=1, random_state=seed).reset_index(drop=True)

	if save_files:
		ordered_filenames_lab_df.to_csv('data/utkface_df.csv', index=False)

	return ordered_filenames_lab_df, length_embeddings


def get_sb_train_test_indexes(diveface_df, seed, length_embedding, spl=0.7):
	train_indexes = []
	test_indexes = []
	random.seed(seed)
	# here we do not need the embeddings
	diveface_df = diveface_df.drop(['f' + str(i) for i in range(length_embedding)], axis=1)
	# to maintain the original indexes
	diveface_df['initialIndex'] = diveface_df.index.values
	# specific for utkface
	for age_code in range(3):
		ref = diveface_df.loc[diveface_df['age'] == age_code]
		indexes = list(ref['initialIndex'])
		random.shuffle(indexes)
		train_indexes += indexes[:int(spl * len(indexes))]
		test_indexes += indexes[int(spl * len(indexes)):]

	random.shuffle(train_indexes)
	return train_indexes, test_indexes


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
