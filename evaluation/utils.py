import matplotlib.pyplot as plt
import numpy as np


key_list = ['first', 'second', 'third']
key2_partial_list = ['sex', 'age', 'ethnicity']


def get_metric_dict():
	metrics = {}
	for key in key_list:
		metrics[key] = {}
		for key2 in ['verification'] + key2_partial_list:
			metrics[key][key2] = []
	return metrics


def store_metrics(metrics, ive_method, sb_metrics, ver_metric):
	key = key_list[ive_method]
	for i, key2 in enumerate(key2_partial_list):
		metrics[key][key2].append(sb_metrics[i])
	metrics[key]['verification'].append(ver_metric)
	return metrics


def plot_metrics(metrics):
	plt.figure()
	plt.xlabel('Epoch')
	x_points = np.arange(1, len(metrics[key_list[0]][key2_partial_list[0]]) + 1)
	for i, key in enumerate(key_list):
		plt.subplot(3, 1, i + 1)
		for key2 in key2_partial_list + ['verification']:
			data = np.array(metrics[key][key2])
			np.save('results/' + key + '_' + key2 + '.npy', data)
			plt.plot(x_points, data, label=key2)
	plt.legend(shadow=True)
	plt.savefig('results/metrics.pdf')


