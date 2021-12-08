import numpy as np
import pandas as pd

def input_label_split(**kwargs):
	filepath = kwargs.get('filepath')
	df = kwargs.get('df')

	if filepath != None:
		df = pd.read_csv(filepath)
	
	headers = df.columns.values.tolist()
	data = df.values
	d = data.shape[1]
	x = data[:, :d-1].reshape(-1, d-1)
	y = data[:, d-1].reshape(-1, 1)
	return x, y, headers

def to_matrix(lst):
	return np.array(lst).reshape(-1, 1)