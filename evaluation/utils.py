import matplotlib.pyplot as plt
import numpy as np
import os
import random


key_list = ['first', 'second', 'third']


def get_labels(db):
	if db == 'diveface':
		return ['sex', 'ethnicity']
	elif db == 'utkface':
		return ['age']


def get_metric_dict(classifiers, db):
	metrics = {}
	for key in key_list:
		metrics[key] = {}
		for key2 in get_labels(db):
			metrics[key][key2] = {}
			for c in classifiers:
				metrics[key][key2][c] = []
		metrics[key]['verification'] = []
	return metrics


def store_metrics(metrics, ive_method, sb_metrics, ver_metric, db):
	key = key_list[ive_method]
	for i, key2 in enumerate(get_labels(db)):
		for c in list(sb_metrics[0].keys()):
			metrics[key][key2][c].append(sb_metrics[i][c])
	metrics[key]['verification'].append(ver_metric)
	return metrics


def plot_metrics(metrics, folder, db, save_files=True):
	verification = not(metrics[key_list[0]]['verification'][0] == 'na')
	plt.figure()
	plt.xlabel('Epoch')
	x_points = np.arange(1, len(metrics[key_list[0]]['verification']) + 1)
	for i, key in enumerate(key_list):
		plt.subplot(3, 1, i + 1)
		for key2 in get_labels(db):
			classifiers = list(metrics[key][key2].keys())
			for ii, c in enumerate(classifiers):
				data = np.array(metrics[key][key2][c])
				if save_files:
					np.save(os.path.join(os.path.join('results', folder), key + '_' + key2 + '_' + c + '.npy'), data)
				if ii == 0:
					# for the moment, plot the scores from only one classifier
					plt.plot(x_points, data, label=key2)
		if verification:
			data = np.array(metrics[key]['verification'])
			if save_files:
				np.save(os.path.join(os.path.join('results', folder), key + '_' + 'verification' + '.npy'), data)
			plt.plot(x_points, data, label='verification')
	plt.legend(shadow=True)
	if verification:
		if save_files:
			plt.savefig(os.path.join(os.path.join('results', folder), 'metrics.pdf'))

	for key2 in get_labels(db) + (['verification'] if verification else []):
		plt.figure()
		plt.xlabel('Epoch')
		x_points = np.arange(1, len(metrics[key_list[0]]['verification']) + 1)
		for key in key_list:
			if not key2 == 'verification':
				# for the moment, plot the scores from only one classifier
				c = list(metrics[key][key2].keys())[0]
				data = np.array(metrics[key][key2][c])
			else:
				data = np.array(metrics[key][key2])
			plt.plot(x_points, data, label=key)
		plt.legend(shadow=True)
		if save_files:
			plt.savefig(os.path.join(os.path.join('results', folder), 'metrics_{}.pdf'.format(key2)))


def create_random_mask_no_pca(seed, length_embedding):
	random.seed(seed)
	num_epochs = 170
	masks = []
	final_folder = os.path.join('results', str(seed) + '_rdm', 'method1')
	os.makedirs(final_folder, exist_ok=True)

	for epoch in range(num_epochs):
		indexes = [ii for ii in range(length_embedding - (epoch * 3))]
		random.shuffle(indexes)
		ind_false = indexes[:3]
		mask = np.array([ix not in ind_false for ix in range(length_embedding - (epoch * 3))])
		masks.append(mask)
		np.save(os.path.join(final_folder, "{0:03}".format(epoch)), mask)

	return masks


# def create_random_masks_pca(seed, k, length_embedding):
# 	random.seed(seed)
# 	masks = {'1': [], '2': [], '3': []}
# 	num_epochs = 170
# 	for i in range(1, 4):
# 		indexes = [ii for ii in range(k, length_embedding)]
# 		random.shuffle(indexes)
# 		for epoch in range(num_epochs):
# 			ind_false = indexes[:(epoch+1)*3]
# 			mask = np.array([ix not in ind_false for ix in range(length_embedding)])
# 			masks[str(i)].append(mask)
# 	return masks
