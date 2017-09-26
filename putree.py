# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import rankdata


class PUTREE():
	def __init__(self, pos_level, max_depth, min_size):
		self.pos_level = pos_level
		self.max_depth = max_depth
		self.min_size = min_size
		self.tree = {}

	# Select the best split point for a dataset
	def get_split(self, dataset, lables):
		unl = max(abs(np.sum((lables-1)/2)), 1e-8)
		pos = max(abs(np.sum((lables+1)/2)), 1e-8)
		sp_index, sp_value, sp_gini = 999, -1e08, 1e08
		for index in range(0, dataset.shape[1]):
			data_sp = dataset[:, index]
			values = np.unique(data_sp)
			for value in values:
				left = np.where(data_sp <= value)[0]
				right = np.where(data_sp > value)[0]
				l_p = max(np.sum((lables[left]+1)/2), 1e-8)
				l_n = max(-np.sum((lables[left]-1)/2), 1e-8)
				r_p = max(np.sum((lables[right]+1)/2), 1e-8)
				r_n = max(-np.sum((lables[right]-1)/2), 1e-8)
				l = left.shape[0]*1.0
				r = right.shape[0]*1.0
				l_p1 = min(1, (l_p/pos)*self.pos_level*(unl/l_n))
				r_p1 = min(1, (r_p/pos)*self.pos_level*(unl/r_n))
				l_gini = 1-(l_p1*l_p1+(1-l_p1)*(1-l_p1))
				r_gini = 1-(r_p1*r_p1+(1-r_p1)*(1-r_p1))
				gini_node = l/(l+r)*l_gini+r/(l+r)*r_gini
				print(index, value, gini_node)
				if gini_node < sp_gini:
					sp_index = index
					sp_value = value
					sp_gini = gini_node
					sp_groups = (dataset[left], dataset[right])
					sp_lables = (lables[left], lables[right])
		return {'index': sp_index, 'value': sp_value, 'groups': sp_groups, 'lables': sp_lables}

	@staticmethod
	# Create a terminal node value
	def to_terminal(lables):
		if np.sum(lables) >= 0:
			return 1
		else:
			return 0

	# Create child splits for a node or make terminal
	def split(self, node,  depth):
		left, right = node['groups']
		left_l, right_l = node['lables']
		del(node['groups'])
		del(node['lables'])
		# check for a no split
		if left.shape[0] == 0 or right.shape[0] == 0:
			node['left'] = node['right'] = self.to_terminal(np.append(left_l, right_l))
			return
		# check for max depth
		if depth >= self.max_depth:
			node['left'], node['right'] = self.to_terminal(left_l), self.to_terminal(right_l)
			return
		# process left child
		if left.shape[0] <= self.min_size:
			node['left'] = self.to_terminal(left_l)
		elif left_l.shape[0] == abs(np.sum(left_l)):
			node['left'] = self.to_terminal(left_l)
		else:
			node['left'] = self.get_split(left, left_l)
			self.split(node['left'], depth+1)
		# process right child
		if len(right) <= self.min_size:
			node['right'] = self.to_terminal(right)
		elif right_l.shape[0] == abs(np.sum(right_l)):
			node['right'] = self.to_terminal(right_l)
		else:
			node['right'] = self.get_split(right, right_l)
			self.split(node['right'], depth+1)

	# Build a decision tree
	def fit(self, train, lables):
		root = self.get_split(train, lables)
		self.split(root, 1)
		self.tree = root

	# Print a decision tree
	def print_tree(self, node=0, depth=0):
		if depth == 0:
			node = self.tree
		if isinstance(node, dict):
			print('%s[X%d <= %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
			self.print_tree(node['left'], depth+1)
			self.print_tree(node['right'], depth+1)
		else:
			print('%s[%s]' % ((depth*' ', node)))

	def predict_x(self, node, row):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):
				return self.predict_x(node['left'], row)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self.predict_x(node['right'], row)
			else:
				return node['right']

	def predict(self, xs):
		y = np.zeros(xs.shape[0])
		for i in range(0, xs.shape[0]):
			y[i] = self.predict_x(self.tree, xs[i, :])
		return y

	def etk(self, x_test, y_test):
		y = self.predict(x_test)
		n_pos = np.where(y_test == 1)[0].__len__()*1.0
		n_unl = np.where(y_test == -1)[0].__len__()*1.0
		pos_np = np.where((y-y_test) == -1)[0].__len__()*1.0
		unl_pos = np.where((y-y_test) == 2)[0].__len__()*1.0
		tk = pos_np/n_pos+unl_pos/n_unl
		return tk


class PURF():
	def __init__(self, max_depth, min_size, n_tree=10, boot_r=0.7):
		self.n_tree = n_tree
		self.forest = []
		self.boot_r =boot_r
		self.max_depth = max_depth
		self.min_size = min_size

	def fit(self, x_train, y_train):
		for i in range(0, self.n_tree):
			tree = self.get_trees(x_train, y_train)
			self.forest.append(tree)

	def bootstrap(self, y_train):
		pos = np.where(y_train == 1)[0]
		unls = np.where(y_train == -1)[0]
		pos = np.random.shuffle(pos)
		unls = np.random.shuffle(unls)
		train = np.append(pos[:np.floor(self.boot_r*pos.__len__())], unls[:np.floor(self.boot_r*unl.__len__())])
		val = np.append(pos[np.floor(self.boot_r*pos.__len__()):], unls[np.floor(self.boot_r*unl.__len__()):])
		return (train, val)

	def get_trees(self, x_train, y_train):
		train_id, val_id = self.bootstrap(y_train)
		train_x = x_train[train_id]
		train_y = y_train[train_id]
		val_x = x_train[val_id]
		val_y = y_train[val_id]
		best_etk = 10
		for j in range(1, 10):
			pos_level = 1.0*j/10
			pu_tree = PUTREE(pos_level, self.max_depth, self.min_size)
			pu_tree.fit(train_x, train_y)
			etk = pu_tree.etk(val_x, val_y)
			if etk < best_etk:
				best_etk = etk
				final_tree = pu_tree
		return final_tree

	def predict(self, x_pred):
		pass





dataset = np.array([[2.771244718,1.784783929, -1],
	[1.728571309, 1.169761413, -1],
	[3.678319846, 2.81281357, -1],
	[3.961043357, 2.61995032,-1],
	[2.999208922,2.209014212,-1],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]])

pu_tree = PUTREE(0.5, 5, 1)
pu_tree.fit(dataset, dataset[:, -1])
pu_tree.print_tree()
y = pu_tree.predict(dataset)
etk = pu_tree.etk(dataset, dataset[:, -1])
print etk
print(y)

