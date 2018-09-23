# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp
import pandas as pd
import sys
import time
from scipy.stats import rankdata



class PUTREE():
	def __init__(self, pos_level, max_depth, min_size):
		self.pos_level = pos_level
		self.max_depth = max_depth
		self.min_size = min_size
		self.tree = {}

		# Select the best split point for a dataset
	def get_split(self, dataset, lables, feature_id):
		unl_cnt = max(abs(np.sum((lables-1)/2)), 1e-8)
		pos_cnt = max(abs(np.sum((lables+1)/2)), 1e-8)
		total_cnt = unl_cnt+pos_cnt
		pos = (lables+1)/2
		unl = -(lables-1)/2
		p_factor = 1.0*(unl_cnt/pos_cnt)*self.pos_level
		sp_index, sp_value, sp_gini = 999, -1e08, 1e08
		for index in feature_id:
			data_sp = dataset[:, index]
			data_df = pd.DataFrame({'data': data_sp, 'pos': pos, 'unl': unl, 'lable': lables})
			f = {'pos': 'sum', 'unl': 'sum', 'lable': 'count'}
			l_gini_df = data_df.groupby('data').agg(f).sort_index(ascending=True).cumsum()
			r_gini_df = data_df.groupby('data').agg(f).sort_index(ascending=False).cumsum()
			gini_df = l_gini_df.join(r_gini_df, how='left', lsuffix='_l', rsuffix='_r')
			gini_df['lable_r'] = total_cnt-gini_df['lable_l']
			del data_df, l_gini_df, r_gini_df
			gini_df['p1_l'] = 1.0*gini_df['pos_l']/gini_df['unl_l']*p_factor
			gini_df['p1_r'] = 1.0*gini_df['pos_r']/gini_df['unl_r']*p_factor
			gini_df.loc[gini_df['p1_l'] > 1, 'p1_l'] = 1
			gini_df.loc[gini_df['p1_r'] > 1, 'p1_r'] = 1
			gini_df['gini_l'] = 1-(gini_df['p1_l']*gini_df['p1_l']+(1-gini_df['p1_l'])*(1-gini_df['p1_l']))
			gini_df['gini_r'] = 1-(gini_df['p1_r']*gini_df['p1_r']+(1-gini_df['p1_r'])*(1-gini_df['p1_r']))
			gini_df['gini'] = gini_df['gini_l']*gini_df['lable_l']/total_cnt+gini_df['gini_r']*gini_df['lable_r']/total_cnt
			min_gini = gini_df['gini'].min()
			if min_gini < sp_gini:
				sp_gini = min_gini
				sp_value = gini_df['gini'].idxmin()
				sp_index = index
		left = np.where(dataset[:, sp_index] <= sp_value)
		right = np.where(dataset[:, sp_index] > sp_value)
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
	def split(self, node,  depth, feature_id):
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
			node['left'] = self.get_split(left, left_l, feature_id)
			self.split(node['left'], depth+1, feature_id)
		# process right child
		if len(right) <= self.min_size:
			node['right'] = self.to_terminal(right)
		elif right_l.shape[0] == abs(np.sum(right_l)):
			node['right'] = self.to_terminal(right_l)
		else:
			node['right'] = self.get_split(right, right_l, feature_id)
			self.split(node['right'], depth+1, feature_id)

	# Build a decision tree
	def fit(self, train, lables, feature_id=None):
		if feature_id == None:
			feature_id = range(0, train.shape[1])
		root = self.get_split(train, lables, feature_id)
		self.split(root, 1, feature_id)
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

	def go_node(self, node, xs, indices):
		left = np.where(xs[indices, node['index']] <= node['value'])[0]
		if isinstance(node['left'], dict):
			self.go_node(node['left'], xs, indices[left])
		else:
			self.y_pred[indices[left]] = node['left']
		right = np.where(xs[indices, node['index']] > node['value'])[0]
		if isinstance(node['right'], dict):
			self.go_node(node['right'], xs, indices[right])
		else:
			self.y_pred[indices[right]] = node['right']

	def predict(self, xs):
		self.y_pred = np.ones(xs.shape[0])*-1
		indices = np.array(range(0, xs.shape[0]))
		self.go_node(self.tree, xs, indices)
		y = self.y_pred.copy()
		del self.y_pred
		return y

	def etk(self, x_test, y_test):
		y = self.predict(x_test)
		n_pos = np.where(y_test == 1)[0].__len__()*1.0
		n_unl = np.where(y_test == -1)[0].__len__()*1.0
		pos_np = np.where((y-y_test) == -1)[0].__len__()*1.0
		unl_pos = np.where((y-y_test) == 2)[0].__len__()*1.0
		tk = pos_np/n_pos+unl_pos/n_unl
		return tk


def parallel_call(params):  # a helper for calling 'remote' instances
	cls = getattr(sys.modules[__name__], params[0])  # get our class type
	instance = cls.__new__(cls)  # create a new instance without invoking __init__
	instance.__dict__ = params[1]  # apply the passed state to the new instance
	method = getattr(instance, params[2])  # get the requested method
	args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
	return method(*args)  # expand arguments, call our method and return the result

class PURF(object):
	def __init__(self, max_depth, min_size, n_tree=10, boot_r=0.7, n_jobs=2, max_features=None, random_state=0):
		self.n_tree = n_tree
		self.forest = []
		self.boot_r = boot_r
		self.max_depth = max_depth
		self.min_size = min_size
		cpu_cnt = mp.cpu_count()
		self.n_jobs = min(n_jobs, n_tree, cpu_cnt)
		self.max_features = max_features
		self.random_state = random_state

	def fit(self, x_train, y_train):
		# try multi_process
		fit_args = []
		for i in range(0, self.n_tree):
			fit_args.append((x_train, y_train, i))
		p = mp.Pool(processes=self.n_jobs)
		result = p.map(parallel_call, self.prepare_call("get_trees", fit_args))
		p.close()
		# p.join()
		print(result)
		self.forest = result
		# self.forest = [res.get() for res in result]

		# # normal
		# for i in range(0, self.n_tree):
		# 	tree = self.get_trees(x_train, y_train, i)
		# 	self.forest.append(tree)

	def prepare_call(self, name, args):  # creates a 'remote call' package for each argument
		for arg in args:
			yield [self.__class__.__name__, self.__dict__, name, arg]

	def bootstrap(self, x_train, y_train, random_state):
		pos = np.where(y_train == 1)[0]
		unl = np.where(y_train == -1)[0]
		np.random.seed(random_state)
		np.random.shuffle(pos)
		np.random.shuffle(unl)
		train_id = np.append(pos[:np.int(self.boot_r*pos.__len__())], unl[:np.int(self.boot_r*unl.__len__())])
		val_id = np.append(pos[np.int(self.boot_r*pos.__len__()):], unl[np.int(self.boot_r*unl.__len__()):])

		feature_len = min(self.max_features, x_train.shape[1])
		features = range(0, x_train.shape[1])
		np.random.shuffle(features)
		feature_id = features[: feature_len]
		return train_id, val_id, feature_id

	def get_trees(self, x_train, y_train, random_state):
		train_id, val_id, feature_id = self.bootstrap(x_train, y_train, random_state)
		train_x = x_train[train_id]
		train_y = y_train[train_id]
		val_x = x_train[val_id]
		val_y = y_train[val_id]
		best_etk = 10
		for j in range(1, 10):
			pos_level = 1.0*j/10
			pu_tree = PUTREE(pos_level, self.max_depth, self.min_size)
			pu_tree.fit(train_x, train_y, feature_id)
			etk = pu_tree.etk(val_x, val_y)
			if etk < best_etk:
				best_etk = etk
				final_tree = pu_tree
		return final_tree

	def predict(self, x_pred):
		y_preds = []
		for tree in self.forest:
			y_preds.append(tree.predict(x_pred))
		y_preds_array = np.array(y_preds).transpose()
		y_pred = (1.0*np.sum(y_preds_array, axis=1)/self.n_tree >= 0.5)
		return y_pred








dataset = np.array([[2.771244718,1.784783929, -1],
	[1.728571309, 1.169761413, -1],
	[3.678319846, 2.81281357, -1],
	[3.961043357, 2.61995032,-1],
	[2.999208922,2.209014212,-1],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,-1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]])
#
# pu_tree = PUTREE(0.9, 5, 1)
# pu_tree.fit(dataset, dataset[:, -1])
# pu_tree.print_tree()
# y = pu_tree.predict(dataset)
# etk = pu_tree.etk(dataset, dataset[:, -1])
# print etk
# print(y)

# purf = PURF(max_depth=1e08, min_size=1, n_tree=10, boot_r=0.7, n_jobs=2, max_features=2, random_state=0)
# purf.fit(dataset, dataset[:, -1])
# y = purf.predict(dataset)
# print(y)
print(time.asctime(time.localtime(time.time())))
if __name__ == "__main__":
	purf = PURF(max_depth=1e08, min_size=1, n_tree=10, boot_r=0.7, n_jobs=3, max_features=2, random_state=0)
	purf.fit(dataset, dataset[:, -1])
	y = purf.predict(dataset)
	print(y)
print(time.asctime(time.localtime(time.time())))