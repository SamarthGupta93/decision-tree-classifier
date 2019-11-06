import pandas as pd
import numpy as np
from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split

columns=['Uniformity of Cell Size','Uniformity of Cell Shape','Label']

def run():
	df = pd.read_csv('breast-cancer.data.txt',names=columns)
	X = df.iloc[:,:2]
	y = df.iloc[:,2]
	X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)
	decision_tree = DecisionTree(min_samples=15)
	# Build Tree
	decision_tree.fit(X_train,y_train)
	# Print Tree
	decision_tree.print_tree()
	# Predict
	print("----- Prediction -----")
	print("Label: {}".format(decision_tree.predict(np.array([10,10]))))
	# Calculate accuracy on test set
	print("\n----- Accuracy -----")
	print("Accuracy: {}".format(decision_tree.accuracy_score(X_test,y_test)))


if __name__=="__main__":
	run()