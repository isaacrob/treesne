import numpy as np
import codecs
import json
import tensorflow as tf
from sklearn.datasets import make_swiss_roll
import scipy

def load_json_files(file_path):
	'''
	Credit: CPSC 453 assignments, Scott Gigante (TA)
	Loads data from a json file
	Inputs:
		file_path   the path of the .json file that you want to read in
	Outputs:
		my_array	this is a numpy array if data is numeric, it's a list if it's a string
	'''

	#  load data from json file
	with codecs.open(file_path, 'r', encoding='utf-8') as handle:
		json_data = json.loads(handle.read())

	# if a string, then returns list of strings
	if not isinstance(json_data[0], str):
		# otherwise, it's assumed to be numeric and returns numpy array
		json_data = np.array(json_data)

	return json_data

def load_coil_20(file_path = "/Users/isaac/Downloads/COIL20.mat"):
	data = scipy.io.loadmat(file_path)
	X = data['X']
	target = data['Y'].reshape(-1)

	return X, target

def load_coil_100(file_path = "/Users/isaac/Downloads/COIL100.mat"):
	data = scipy.io.loadmat(file_path)
	# print(data)
	X = data['fea']
	target = data['gnd'].reshape(-1)

	return X, target

def load_USPS(file_path = "/Users/isaac/Downloads/USPS.mat"):
	data = scipy.io.loadmat(file_path)
	# print(data)
	X = data['fea']
	target = data['gnd'].reshape(-1)
	target = target - 1

	return X, target

def load_big_mnist(train = False):
	"""
	Loads the large MNIST dataset from keras 
	and returns the data and labels for the test dataset (10k observations)
	"""
	mnist = tf.keras.datasets.mnist
	(Xt, ttarget), (X, target) = mnist.load_data()
	Xt, X = Xt / 255.0, X / 255.0

	if train:
		X = Xt.reshape(Xt.shape[0], -1)
		target = ttarget
	else:
		X = X.reshape(X.shape[0], -1)
	return X, target

def load_cytof():
	cytof = load_json_files("../cytof_data/cytof_data.json")
	channels = load_json_files("../cytof_data/cytof_channel_names.json")

	highest_expressed = get_labels_by_max(cytof, channels)

	return cytof, channels, highest_expressed

def get_labels_by_max(data, colnames, colnames_subset = None):
	if colnames_subset is not None:
		colnames_subset_set = set(colnames_subset)
		colnums = [i for i in range(len(colnames)) if colnames[i] in colnames_subset_set]
		data = data[:,colnums]
		colnames = colnames_subset

	# scale columns with normal dist
	means = data.mean(axis=0)
	stds = data.std(axis=0)
	stds = np.where(stds==0, 1, stds) #if sd is 0 make it 1 instead so don't divide by 0
	scaled = (data - means)/stds

	max_cols = np.argmax(scaled, axis=1)
	max_labels = list(np.asarray(colnames)[max_cols])

	return max_labels



