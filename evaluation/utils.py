import matplotlib.pyplot as plt
import numpy as np
import os


key_list = ['first', 'second', 'third']
key2_partial_list = ['sex', 'age', 'ethnicity']


def get_metric_dict(classifiers):
	metrics = {}
	for key in key_list:
		metrics[key] = {}
		for key2 in key2_partial_list:
			metrics[key][key2] = {}
			for c in classifiers:
				metrics[key][key2][c] = []
		metrics[key]['verification'] = []
	return metrics


def store_metrics(metrics, ive_method, sb_metrics, ver_metric):
	key = key_list[ive_method]
	for i, key2 in enumerate(key2_partial_list):
		for c in list(sb_metrics[0].keys()):
			metrics[key][key2][c].append(sb_metrics[i][c])
	metrics[key]['verification'].append(ver_metric)
	return metrics


def plot_metrics(metrics, seed, save_files=True):
	seed = str(seed)
	os.makedirs("results", exist_ok=True)
	os.makedirs(os.path.join('results', seed), exist_ok=True)
	plt.figure()
	plt.xlabel('Epoch')
	x_points = np.arange(1, len(metrics[key_list[0]]['verification']) + 1)
	for i, key in enumerate(key_list):
		plt.subplot(3, 1, i + 1)
		for key2 in key2_partial_list:
			classifiers = list(metrics[key][key2].keys())
			for ii, c in enumerate(classifiers):
				data = np.array(metrics[key][key2][c])
				if save_files:
					np.save(os.path.join(os.path.join('results', seed), key + '_' + key2 + '_' + c + '.npy'), data)
				if ii == 0:
					# for the moment, plot the scores from only one classifier
					plt.plot(x_points, data, label=key2)
		data = np.array(metrics[key]['verification'])
		if save_files:
			np.save(os.path.join(os.path.join('results', seed), key + '_' + 'verification' + '.npy'), data)
		plt.plot(x_points, data, label='verification')
	plt.legend(shadow=True)
	if save_files:
		plt.savefig(os.path.join(os.path.join('results', seed), 'metrics.pdf'))

	for key2 in key2_partial_list + ['verification']:
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
			plt.savefig(os.path.join(os.path.join('results', seed), 'metrics_{}.pdf'.format(key2)))


