import tqdm
import os
import numpy as np
import random
import pandas as pd
import random

# DiveFace data are divided in six folders (demographic groups)
folders = ['AM4K', 'AW4K', 'BM4K', 'BW4K', 'CM4K', 'CW4K' ]


def _get_label_coded(label):
	sex = 0 if label[1] == 'M' else 1
	eth = 0 if label[0] == 'A' else 1 if label[0] == 'B' else 2
	return [sex, None, eth]


def get_embeddings(embeddings_path):
	embeddings = []
	labels = []
	filenames = []
	users = []
	# return embeddings and the list of filenames, in the same df
	for path, subdirs, files in tqdm.tqdm(os.walk(embeddings_path)):
		for name in [f for f in files if f.find('.npy') >= 0]:
			# store embeddings normalized with L2-norm
			embedding = np.load(os.path.join(path, name))
			embedding = embedding / np.linalg.norm(embedding, ord=2)
			embeddings.append(embedding)
			# get the user name
			user = path.split('\\')[-1]
			users.append(user)
			# get the labels (contained in the name of the second last folder)
			label = _get_label_coded(path.split('\\')[-2])
			labels.append(label)
			# remove file extension (npy)
			filename = name[:-4]
			filenames.append(filename)

	# create DataFrame with filenames, embeddings, users, and files
	embeddings = np.array(embeddings)
	files_embeddings_df = pd.DataFrame(embeddings, columns=['f'+str(i) for i in range(len(embedding))])
	files_embeddings_df['filename'] = filenames
	files_embeddings_df['users'] = users

	labels = np.array(labels)
	labels_df = pd.DataFrame(labels, columns=['sex', 'age', 'ethnicity'])
	files_embeddings_df = pd.concat([files_embeddings_df, labels_df], axis=1)

	return files_embeddings_df, len(embedding)


def get_diveface_df(embeddings_path, seed, save_files=False, limit_size=False):
	ordered_filenames_lab_df, length_embeddings = get_embeddings(embeddings_path)
	if limit_size:
		random.seed(seed)
		pats_to_keep = []
		# specific for diveface
		for sex_code in range(2):
			for eth_code in range(3):
				tmp = ordered_filenames_lab_df.loc[(ordered_filenames_lab_df['sex'] == sex_code) &
												   (ordered_filenames_lab_df['ethnicity'] == eth_code)]
				pats = list(tmp['users'].unique())
				random.shuffle(pats)
				pats_to_keep += pats[:500]

		ordered_filenames_lab_df = ordered_filenames_lab_df[ordered_filenames_lab_df['users'].isin(pats_to_keep)]

	ordered_filenames_lab_df = ordered_filenames_lab_df.sample(frac=1, random_state=seed).reset_index(drop=True)

	if save_files:
		ordered_filenames_lab_df.to_csv('data/diveface_df.csv', index=False)

	return ordered_filenames_lab_df, length_embeddings


def get_sb_train_test_indexes(diveface_df, seed, length_embedding, spl=0.7):
	train_indexes = []
	test_indexes = []
	random.seed(seed)
	# here we do not need the embeddings
	diveface_df = diveface_df.drop(['f' + str(i) for i in range(length_embedding)], axis=1)
	# to maintain the original indexes
	diveface_df['initialIndex'] = diveface_df.index.values
	# consider one random sample for each subject
	df_gby = diveface_df.groupby('users').apply(lambda x: x.sample(1, random_state=seed)).reset_index(drop=True)
	# specific for diveface
	for sex_code in range(2):
		for eth_code in range(3):
			ref = df_gby.loc[(df_gby['sex']==sex_code) & (df_gby['ethnicity']==eth_code)]
			indexes = list(ref['initialIndex'])
			random.shuffle(indexes)
			train_indexes += indexes[:int(spl * len(indexes))]
			test_indexes += indexes[int(spl * len(indexes)):]

	random.shuffle(train_indexes)
	return train_indexes, test_indexes


def get_verification_indexes(diveface_df, seed, length_embedding, genuine=3):
	dict_verification_indexes = {}
	random.seed(seed)
	# here we do not need the embeddings
	diveface_df = diveface_df.drop(['f' + str(i) for i in range(length_embedding)], axis=1)
	# consider three random sample for each subject
	df_gby = diveface_df.groupby('users').apply(lambda x: x.sample(min(3, len(x)), random_state=seed))

	for ax, _ in df_gby.iterrows():
		user = ax[0]
		initial_index = ax[1]
		if user not in list(dict_verification_indexes.keys()):
			dict_verification_indexes[user] = []
		dict_verification_indexes[user].append(initial_index)

	return dict_verification_indexes


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
