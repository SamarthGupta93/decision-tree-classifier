import math
import numpy as np

CATEGORICAL='C'
NUMERICAL='N'

class TreeNode():

	def __init__(self,depth,X,y,row_idxs,class_distribution,impurity,min_leaf_samples,categorical_features=[]):
		self.depth = depth	# Depth of the Node. Can be used as a stopping criteria for the tree. Not configured currently
		self.x = X 		# Feature matrix
		self.y = y 		# Labels
		self.row_idxs = row_idxs 	# Rows corresponding to this Node
		self.row_count = len(row_idxs) 	# Number of data points
		self.col_count = self.x.shape[1]	# Number of feature dimensions
		self.class_distribution = class_distribution 	# Class distribution in this node
		self.question = {'col':None,'value':None}	# Question to be asked in this node for splitting
		self.impurity = impurity 	# Impurity (entropy) of the node
		self.is_leaf = True		# Default is kept as True. Changes to False if a better split is found
		self.categorical_features = categorical_features	# Categorical column indexes in the features
		self.min_leaf_samples = min_leaf_samples 	# STOPPING CRITERIA: Minimum number of samples that should be present in the leaves. 	
		self.left = None	# Left child of the node
		self.right = None	# Right child of the node
		self.split_node()	# Find a feature split for the node, if exists.
		if self.is_leaf: self.find_major_class() 	# If leaf, get the major class label. This will be the class label for any data point that reaches this leaf.


	def split_node(self):
		# No Need to split if the impurity is 0
		if self.impurity==0: return
		# Find best (column,split_value) pair. This will be the question to ask at this node
		self.find_best_split()
		if self.is_leaf: return
		left_idxs = np.where(self.x[self.row_idxs,self.question['col']]<=self.question['value'])[0]
		right_idxs = np.where(self.x[self.row_idxs,self.question['col']]>self.question['value'])[0]
		self.left = TreeNode(self.depth+1,self.x,self.y,self.row_idxs[left_idxs],self.left_distribution,self.left_impurity,self.min_leaf_samples)
		self.right = TreeNode(self.depth+1,self.x,self.y,self.row_idxs[right_idxs],self.right_distribution,self.right_impurity,self.min_leaf_samples)


	def find_best_split(self):
		for col in range(self.col_count):
			x_col = self.x[self.row_idxs,col]
			splits = np.unique(x_col)
			# If column is categorical, loop over all the subsets
			if col in self.categorical_features:
				col_type = CATEGORICAL
				splits = get_subsets(unique_values)
			else: 
				col_type = NUMERICAL
				splits.sort()

			# Loop over all the splits and select the best split condition for the column
			for i in range(len(splits)):
				if col_type==CATEGORICAL: split_value = splits[i]
				else: split_value = (splits[i]+splits[i+1])/2 if i<len(splits)-1 else splits[i]
				# For every split, calculate impurity. The best split will be the (column,split_value) pair with lowest impurity
				label_proportion_left, left_total, label_proportion_right, right_total = self.feature_split(col,split_value,col_type)
				# Check if split passes the stopping rule
				if left_total<self.min_leaf_samples or right_total<self.min_leaf_samples: continue
				# Calculate impurity of split
				entropy_left = self.entropy(label_proportion_left,left_total)
				entropy_right = self.entropy(label_proportion_right,right_total)
				impurity = self.weighted_entropy(entropy_left,entropy_right,left_total,right_total)
				# If impurity is minimum, update best column and split value
				if impurity<self.impurity:
					self.update_best_split(impurity,entropy_left,entropy_right,col,split_value)
					self.left_distribution = label_proportion_left
					self.right_distribution = label_proportion_right
					self.is_leaf = False


	def feature_split(self,col,split_value,col_type):
		label_count = {'left':{},'right':{}}
		left_total,right_total = 0,0
		for idx in self.row_idxs:
			x_val = self.x[idx,col]
			y_val = self.y[idx]
			if (col_type==NUMERICAL and x_val<=split_value) or (col_type==CATEGORICAL and x_val in split_value):
				if y_val not in label_count['left']:
					label_count['left'][y_val]=0
				label_count['left'][y_val]+=1
				left_total+=1
			else:
				if y_val not in label_count['right']:
					label_count['right'][y_val]=0
				label_count['right'][y_val]+=1
				right_total+=1

		return (label_count['left'],left_total,label_count['right'],right_total)


	def entropy(self,label_proportion_map,total):
		# Entropy is sum(prob*log(1/prob))
		entropy = 0
		for label,label_count in label_proportion_map.items():
			prob = label_count/total
			entropy+= prob*(math.log(1/prob))
		return entropy


	def weighted_entropy(self,entropy_left,entropy_right,total_left,total_right):
		return (total_left/self.row_count)*entropy_left + (total_right/self.row_count)*entropy_right

	def update_best_split(self,impurity,left_impurity,right_impurity,col_idx,split_value):
		self.impurity = impurity
		self.left_impurity = left_impurity
		self.right_impurity = right_impurity
		self.question['col']=col_idx
		self.question['value']=split_value


	def find_major_class(self):
		max_count = 0
		for label,count in self.class_distribution.items():
			if count>max_count: 
				max_count = count
				self.majority = label


	def print_node(self):
		print(self.depth*'\t'+"Leaf: {}".format(self.is_leaf))
		if not self.is_leaf:
			print(self.depth*'\t'+"Question: X[{}] <= {}".format(self.question['col'],self.question['value']))
		else:
			print(self.depth*'\t'+"Class Distribution {}".format(self.class_distribution))
		

def get_subsets(unique_values):
	result = [[]]
	for value in unique_values:
		subsets_new = []
		for subarr in result:
			subsets_new.append(subarr+[value])
		result+=subsets_new
	return result[1:]	# Returning everything except empty subset ([])