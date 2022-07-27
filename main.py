import os
import data.manage_data_feret as manage_data
import pandas as pd
import IVE.utils as ive_util
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pickle as pk
import tqdm


def zeros_pca(x_total, mask):
	# store indexes where the mask is False.
	indexes = [i for i, m in enumerate(mask) if not m]
	# The features related to those indexes need to be zero.
	for ind in indexes:
		x_total[:, ind] = np.zeros((len(x_total)))
	return x_total


def execute_all(model_feat_importance, transform, seed, blocked_features=0):
	# create
	if not os.path.isfile('data/feret_df.csv'):
		feret_df = manage_data.get_feret_df(r'data/myselection_embeddings', seed, save_files=True)
	else:
		feret_df = pd.read_csv('data/feret_df.csv')

	folder = str(seed) + ('_' + transform)
	if transform == 'pca' or transform == 'ica':
		folder = folder + '_k=' + str(blocked_features)
	os.makedirs("results", exist_ok=True)
	os.makedirs(os.path.join('results', folder), exist_ok=True)

	# set useful variables
	labels = ['sex', 'age', 'ethnicity']
	# train_df contains 4 columns more than the number of features
	length_embeddings = feret_df.shape[1] - 4
	# number of steps and number of eliminations per step have to be adjusted depending on the feature size
	num_steps = 1
	num_eliminations = 3
	num_epochs = 170

	# define classifier
	n_estimators = 30
	model_train = RandomForestClassifier(n_estimators=n_estimators, random_state=seed) if model_feat_importance == 'rf' \
		else ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed) if model_feat_importance == 'et' \
		else GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed) if model_feat_importance == 'gb' \
		else XGBClassifier(n_estimators=n_estimators, random_state=seed) if model_feat_importance == 'xgb' else None

	# set parameters for method one
	n_s = [num_steps, num_steps, num_steps]
	n_e = [int(num_eliminations / 3), int(num_eliminations / 3), int(num_eliminations / 3)]
	elimination_order = [0, 1, 2]

	# get embeddings
	x_total = manage_data.get_x_ready(feret_df, length_embeddings)

	# normalize embeddings
	std_sc = StandardScaler()
	# fit the scaler with only training data
	# TODO: check classes for IVE training
	x_total = std_sc.fit_transform(x_total)
	pk.dump(std_sc, open(os.path.join(os.path.join('results', folder), 'std_sc.pkl'), "wb"))

	# get labels
	y = manage_data.get_y_ready(feret_df, labels)

	if transform == 'pca':
		# transform from original to PCA domain
		pca = PCA(n_components=length_embeddings, random_state=seed)
		x_total = pca.fit_transform(x_total)
		pk.dump(pca, open(os.path.join(os.path.join('results', folder), 'pca.pkl'), "wb"))
	if transform == 'ica':
		# transform from original to ICA domain
		ica = FastICA(n_components=length_embeddings, random_state=seed, whiten='unit-variance', max_iter=1000)
		x_total = ica.fit_transform(x_total)
		pk.dump(ica, open(os.path.join(os.path.join('results', folder), 'ica.pkl'), "wb"))

	x_first = np.copy(x_total)
	x_second = np.copy(x_total)
	x_third = np.copy(x_total)

	for epoch in tqdm.tqdm(range(num_epochs)):
		# print('--- Epoch {} ---'.format(epoch))
		# print('First size: {} - Second size: {} - Third size: {}'.format(x_first.shape[1], x_second.shape[1], x_third.shape[1]))

		first_ive = ive_util.first_method(x_first, y, model_train, n_e, n_s, elimination_order, blocked_features)
		# print('first ive training done')
		second_ive = ive_util.second_method(x_second, y, model_train, num_eliminations, num_steps, blocked_features)
		# print('second ive training done')
		third_ive = ive_util.third_method(x_third, y, model_train, num_eliminations, num_steps, blocked_features)
		# print('third ive training done')

		# get the masks
		first_mask = [f.get_mask() for f in first_ive]
		second_mask = second_ive.get_mask()
		third_mask = third_ive.get_mask()

		# fix the first mask, simplify the representation for the other masks and save them
		ive_util.fix_mask(first_mask, 1, epoch, folder, save=True)
		ive_util.fix_mask(second_mask, 2, epoch, folder, save=True)
		ive_util.fix_mask(third_mask, 3, epoch, folder, save=True)

		# apply IVE
		x_first = first_ive[2].transform(first_ive[1].transform(first_ive[0].transform(x_first)))
		x_second = second_ive.transform(x_second)
		x_third = third_ive.transform(x_third)

	print('Done! folder: ', folder)


for seed in range(10):
	execute_all('rf', 'NO_pca', seed)
	for k in [0, 3, 5]:
		execute_all('rf', 'pca', seed, k)

# for seed in range(10):
# 	execute_all('rf', 'ica', seed)
