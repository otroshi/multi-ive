import numpy as np
import matplotlib.pyplot as plt
import os


base_path = 'results'
seed = [str(i) for i in range(10)]
mode = ['NO_pca'] + ['pca_k=' + str(i) for i in [0, 3, 5]] + ['ica_k=0']
method = ['first', 'second', 'third']
soft_biom = ['age', 'sex', 'ethnicity', 'verification']
classifier = ['mlp', 'svm_lin', 'et', 'log_reg']


dict_all = {}
for s in seed:
	dict_all[s] = {}
	for m in mode:
		dict_all[s][m] = {}
		for meth in method:
			dict_all[s][m][meth] = {}
			for sb in soft_biom:
				dict_all[s][m][meth][sb] = {}
				if not sb == 'verification':
					for c in classifier:
						dict_all[s][m][meth][sb][c] = {}


def add_metrics(dict_all, k1, k2, k3, k4, k5=None):
	folder = k1 + '_' + k2
	file = k3 + '_' + k4 + ('_' + k5 + '.npy' if not k4 == 'verification' else '.npy')
	filename = os.path.join(base_path, folder, file)
	metrics = np.load(filename)
	if len(metrics) > 35:
		index_to_fix = [np.arange(0, 170, 5)]
		metrics = metrics[index_to_fix]
	if not k4 == 'verification':
		dict_all[k1][k2][k3][k4][k5] = metrics
	else:
		dict_all[k1][k2][k3][k4] = metrics


def select_metrics(dict_all, k1, k2, k3, k4, k5=None, fn=None, plot=True):
	metrics = []
	leg = []
	for kk1 in k1:
		for kk2 in k2:
			for kk3 in k3:
				for kk4 in k4:
					if not kk4 == 'verification':
						for kk5 in k5:
							metrics.append(dict_all[kk1][kk2][kk3][kk4][kk5])
							leg.append(kk1 + '_' + kk2 + '_' + kk3 + '_'+ kk4 + '_'+ kk5)
					else:
						metrics.append(dict_all[kk1][kk2][kk3][kk4])
						leg.append(kk1 + '_' + kk2 + '_' + kk3 + '_' + kk4)
	if not plot:
		return metrics, leg
	else:
		plot_selected(metrics, leg, fn)


def plot_selected(metrics, legends, fn, save_files=True):
	# hard coded in case of saving all the epochs
	plt.figure()
	plt.xlabel('Epoch')
	x_points = np.arange(0, len(metrics[0])*5, 5)
	for m, l in zip(metrics, legends):
		plt.plot(x_points, m, label=l)

	plt.legend(shadow=True)
	if save_files:
		plt.savefig(os.path.join('plot', fn + '.pdf'))


def average(dict_all, k2, k3, k4, k5=None):
	data = []
	for s in [str(ss) for ss in range(10)]:
		data.append(dict_all[s][k2][k3][k4][k5] if not k4 == 'verification' else dict_all[s][k2][k3][k4])
	data = np.array(data)
	return data.mean(axis=0), data.std(axis=0)

# load all the metrics (for the moment only related to seed 0)
for s in seed:
	for m in mode:
		for meth in method:
			for sb in soft_biom:
				if not sb == 'verification':
					for c in classifier:
						add_metrics(dict_all, s, m, meth, sb, c)
				else:
					add_metrics(dict_all, s, m, meth, sb)

for key in ['avg', 'std']:
	dict_all[key] = {}
	for m in mode:
		dict_all[key][m] = {}
		for meth in method:
			dict_all[key][m][meth] = {}
			for sb in soft_biom:
				dict_all[key][m][meth][sb] = {}
				if not sb == 'verification':
					for c in classifier:
						dict_all[key][m][meth][sb][c] = {}

for m in mode:
	for meth in method:
		for sb in soft_biom:
			if not sb == 'verification':
				for c in classifier:
					mean, std = average(dict_all, m, meth, sb, c)
					dict_all['avg'][m][meth][sb][c] = mean
					dict_all['std'][m][meth][sb][c] = std
			else:
				mean, std = average(dict_all, m, meth, sb)
				dict_all['avg'][m][meth][sb] = mean
				dict_all['std'][m][meth][sb] = std

# select the metrics to plot with list of keys as parameters
for sb in soft_biom:
	select_metrics(dict_all, ['avg'], ['NO_pca'], method, [sb], ['mlp'], 'NO_pca_' + sb + '_mlp')

select_metrics(dict_all, ['avg'], mode, ['third'], ['sex', 'verification'], ['mlp'], 'third_sex_mlp_verification')
# select_metrics(dict_all, ['avg'], mode, ['third'], ['verification'], ['mlp'], 'third_verification')

select_metrics(dict_all, ['avg'], ['NO_pca', 'ica_k=0', 'pca_k=3'], ['third'], ['sex'], classifier, 'NO_pca_ica_k=0_pca_k=3_third_sex')
select_metrics(dict_all, ['avg'], ['NO_pca', 'ica_k=0', 'pca_k=3'], ['third'], ['age'], classifier, 'NO_pca_ica_k=0_pca_k=3_third_age')
select_metrics(dict_all, ['avg'], ['NO_pca', 'ica_k=0', 'pca_k=3'], ['third'], ['ethnicity'], classifier, 'NO_pca_ica_k=0_pca_k=3_third_ethnicity')

