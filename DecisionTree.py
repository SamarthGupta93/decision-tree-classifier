import pandas as pd
import numpy as np
from TreeNode import TreeNode

class DecisionTree:

	def __init__(self,min_samples=15):
		self.min_leaf_samples = min_samples


	def fit(self,X,y,categorical_features=[]):
		self.x = X.to_numpy()
		self.y = y.to_numpy()
		self.categorical_features = categorical_features
		class_distribution = self.get_class_distribution()
		self.decision_tree = TreeNode(0,self.x,self.y, np.array(range(X.shape[0])), class_distribution, float('inf'),self.min_leaf_samples,categorical_features)


	def predict(self,X_new,node=None):
		if node is None: node = self.decision_tree
		if node.is_leaf: return node.majority
		if (node.question['col'] not in self.categorical_features and X_new[node.question['col']]<=node.question['value']) \
		or (node.question['col'] in self.categorical_features and X_new[node.question['col']] in node.question['value']):
			return self.predict(X_new,node.left)
		else:
			return self.predict(X_new,node.right)


	def predict_all(self,X_batch):
		result = []
		for X in X_batch:
			result.append(self.predict(X))
		return np.array(result)


	def accuracy_score(self,X_test,y_test):
		X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
		result = self.predict_all(X_test)
		return (result==y_test).mean()


	def print_tree(self,node=None,pos=None):
		if node is None: 
			node = self.decision_tree
			print("----- Printing Decision Tree -----\n")
		if pos=="left":
			print(node.depth*'\t' + "Left: ")
		elif pos=="right":
			print(node.depth*'\t' + "Right: ")
		else:
			print(node.depth*'\t' + "Root: ")
		node.print_node()
		print()
		if node.left is not None: self.print_tree(node.left,'left')
		if node.right is not None: self.print_tree(node.right,'right')


	def get_class_distribution(self):
		unique_elements, count = np.unique(self.y,return_counts=True)
		distribution = {}
		for i in range(len(unique_elements)):
			distribution[unique_elements[i]]=count[i]
		return distribution



