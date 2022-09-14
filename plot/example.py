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


def select_metrics(dict_all, k1, k2, k3, k4, k5=None, fn=None, plot=True, std=False, y_lim=[0, 1], special=[], leg_option=False):
	metrics = []
	leg = []
	stds = []
	for kk1 in k1:
		for kk2 in k2:
			for kk3 in k3:
				for kk4 in k4:
					if not kk4 == 'verification':
						if 'avg' in k5:  # average multiple classifiers
							met_c = []
							std_c = []
							for c in classifier:
								met_c.append(dict_all[kk1][kk2][kk3][kk4][c])
								if std:
									std_c.append(dict_all['std'][kk2][kk3][kk4][c])
							met_c = np.array(met_c)
							met_c = met_c.mean(axis=0)
							if std:
								std_c = np.array(std_c)
								std_c = std_c.mean(axis=0)
							metrics.append(met_c)
							leg.append(kk1 + '_' + kk2 + '_' + kk3 + '_'+ kk4 + '_avg')
							if std:
								stds.append(std_c)
						else:  # keep metrics separated for each classifier
							for kk5 in k5:
								metrics.append(dict_all[kk1][kk2][kk3][kk4][kk5])
								leg.append(kk1 + '_' + kk2 + '_' + kk3 + '_'+ kk4 + '_'+ kk5)
								if std:
									stds.append(dict_all['std'][kk2][kk3][kk4][kk5])

					else:
						metrics.append(dict_all[kk1][kk2][kk3][kk4])
						leg.append(kk1 + '_' + kk2 + '_' + kk3 + '_' + kk4)
						if std:
							stds.append(dict_all['std'][kk2][kk3][kk4])
	if not plot:
		return metrics, leg, stds
	else:
		plot_selected(metrics, leg, fn, stds, y_lim, special, leg_option)


def get_special_values(sb_rdm, label):  # label is random NO_PCA or first_components_PCA
	sb = []
	ver = []
	for seed in range(10):
		sb_rdm_values = np.load(r'results/' + str(seed) + '_' + label + '/' + sb_rdm + '.npy')
		if len(sb_rdm_values) > 35:
			index_to_fix = [np.arange(0, 170, 5)]
			sb_rdm_values = sb_rdm_values[index_to_fix]

		ver_rdm_values = np.load(r'results/' + str(seed) + '_' + label + '/verification.npy')
		if len(ver_rdm_values) > 35:
			index_to_fix = [np.arange(0, 170, 5)]
			ver_rdm_values = ver_rdm_values[index_to_fix]

		sb.append(sb_rdm_values)
		ver.append(ver_rdm_values)

	sb = np.array(sb)
	ver = np.array(ver)
	return sb.mean(axis=0), sb.std(axis=0), ver.mean(axis=0), ver.std(axis=0)


def plot_selected(metrics, legends, fn, stds, y_lim, special, leg_option, save_files=True):
	plt.figure()
	plt.xlabel('Epoch')
	plt.title(fn)
	ax = plt.gca()
	ax.set_ylim(y_lim)

	x_points = np.arange(0, len(metrics[0])*5, 5)
	for i, data_plot in enumerate(zip(metrics, legends)):
		m = data_plot[0]
		l = data_plot[1]
		plt.plot(x_points, m, label=l)
		if len(stds) > 0:
			ci = 1.96 * (stds[i] / np.sqrt(10))
			plt.fill_between(x_points, m - ci, m + ci, alpha=.3)

	for label in ['rdm', 'first_components_PCA', 'first_components_PCA_random_elim']:
		if label in special:
			sb_rdm = legends[0].split('_')[-2]
			sb_mean, sb_std, ver_mean, ver_std = get_special_values(sb_rdm, label)
			plt.plot(x_points, sb_mean, label=label + '_' + sb_rdm + '_mlp')
			ci = 1.96 * (sb_std / np.sqrt(10))
			plt.fill_between(x_points, sb_mean - ci, sb_mean + ci, alpha=.3)
			plt.plot(x_points, ver_mean, label=label + '_verification')
			ci = 1.96 * (ver_std / np.sqrt(10))
			plt.fill_between(x_points, ver_mean - ci, ver_mean + ci, alpha=.3)

	if leg_option == 'out':
		plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", shadow=True)
	elif leg_option == 'in':
		plt.legend(shadow=True)
	if save_files:
		plt.savefig(os.path.join('plot', fn + '.jpg'), bbox_inches="tight")


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

sb = 'sex'
l = 'out'
select_metrics(dict_all, ['avg'], ['NO_pca', 'pca_k=0'], ['first'], [sb, 'verification'], ['mlp'], 'first_' + sb +
   '_mlp_verification_1_rdmpca', std=True, special=['rdm', 'first_components_PCA_random_elim'],
			   leg_option=l)
# select the metrics to plot with list of keys as parameters

# for sb, l in zip(['sex', 'age', 'ethnicity'], ['out', '', '']):
# 	select_metrics(dict_all, ['avg'], ['NO_pca', 'ica_k=0'], ['first'], [sb, 'verification'], ['mlp'], 'first_' + sb +
# 				   '_mlp_verification_1', std=True, special=['rdm', 'first_components_PCA'], leg_option=l)
# 	select_metrics(dict_all, ['avg'], ['pca_k=' + str(i) for i in [0, 3, 5]], ['first'], [sb, 'verification'],
# 				   ['mlp'], 'first_' + sb + '_mlp_verification_2', std=True, special=['first_components_PCA'], leg_option=l)
# 	select_metrics(dict_all, ['avg'], ['pca_k=0'], method, [sb, 'verification'],
# 				   ['mlp'], sb + '_mlp_verification_pca_k=0', std=True, leg_option=l)
# 	select_metrics(dict_all, ['avg'], ['pca_k=5'], method, [sb, 'verification'],
# 				   ['mlp'], sb + '_mlp_verification_pca_k=5', std=True, leg_option=l)
#
# 	for c in classifier:
# 		select_metrics(dict_all, ['avg'], ['pca_k=0'], ['third'], [sb], classifier, 'third_' + sb + '_pca_k=0', std=True, leg_option=l)
# 		select_metrics(dict_all, ['avg'], ['pca_k=5'], ['third'], [sb], classifier, 'third_' + sb + '_pca_k=5', std=True, leg_option=l)
