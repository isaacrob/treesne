import numpy as np
import codecs
import json
import tensorflow as tf

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


def load_big_mnist():
	"""
	Loads the large MNIST dataset from keras 
	and returns the data and labels for the test dataset (10k observations)
	"""
	mnist = tf.keras.datasets.mnist
	(Xt, ttarget), (X, target) = mnist.load_data()
	Xt, X = Xt / 255.0, X / 255.0
	X = X.reshape(X.shape[0], -1)
	return X, target

def load_cytof():
	cytof = load_json_files("../cytof_data/cytof_data.json")
	channels = load_json_files("../cytof_data/cytof_channel_names.json")

	# scale cytof columns with normal dist
	cytof_mean = cytof.mean(axis=0)
	cytof_std = cytof.std(axis=0)
	cytof_std = np.where(cytof_std==0, 1, cytof_std) #if sd is 0 make it 1 instead so don't divide by 0
	cytof_scaled = (cytof - cytof_mean)/cytof_std

	highest_expressed_cols = np.argmax(cytof_scaled, axis=1)
	highest_expressed = list(np.asarray(channels)[highest_expressed_cols])

	return cytof, channels, highest_expressed





