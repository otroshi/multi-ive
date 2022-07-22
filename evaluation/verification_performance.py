import numpy as np
from sklearn import metrics
import random

def get_indexes(test_df):
	# return a dict that contains for each patient the indexes of samples to consider for verification
	patients = [p[:5] for p in list(test_df['filename'])]
	dict_ = {}
	for i, p in enumerate(patients):
		if p not in list(dict_.keys()):
			dict_[p] = []
		if len(dict_[p]) < 2:
			dict_[p].append(i)
	return dict_


def create_verification_dict(indexes, test_data):
	dict_ = {}
	for user, indexes in indexes.items():
		dict_[user] = []
		for index in indexes:
			row = test_data[index]
			dict_[user].append(row)
	return dict_


def similarity_score(enrollment, probe, ord=2):
	return np.linalg.norm((enrollment - probe).flatten(), ord=ord) * (-1)


def perform_genuine_comparisons(dict_samples):
	scores = []
	for k, v in dict_samples.items():
		for i in range(len(v)):
			for j in range(i+1, len(v)):
				# consider all the possible pairs only once
				scores.append(similarity_score(v[i], v[j]))
	return np.array(scores)


def perform_impostor_comparisons(dict_samples, seed):
	random.seed(seed)
	# use the first sample for each subject
	scores = []
	# consider all the possible pairs of enrolment - probe
	for k1, v1 in dict_samples.items():
		others = list(dict_samples.keys())
		random.shuffle(others)
		for o in others[:10]:
			# to avoid impostor comparisons from the same subject
			if not k1 == o:
				scores.append(similarity_score(v1[0], dict_samples[o][0]))
	return np.array(scores)


def evaluate_verification(indexes, data, seed):
	dict_ = create_verification_dict(indexes, data)
	genuines = perform_genuine_comparisons(dict_)
	impostors = perform_impostor_comparisons(dict_, seed)

	scores = np.concatenate((genuines, impostors))
	true_labels = [1 for _ in range(len(genuines))] + [0 for _ in range(len(impostors))]
	fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores)

	eer = (fpr[np.argmin(np.absolute(fpr - (1 - tpr)))] + (1 - tpr[np.argmin(np.absolute(fpr - (1 - tpr)))])) / 2
	return eer


