import pandas as pd 
import numpy as np
#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return float((predictions == labels).sum()) / labels.size

#===========================================================================================================

class KNN:
	columns = []
	distance_columns = []
	training = []
	test_data = []
	labels = []
	test_labels = []
	training_labels = []

	def __init__(self):
		self.columns = dataset.columns.values
		self.distance_columns = self.columns[2:]
		encodedData = encodeData(dataset,self.columns[5:len(self.columns)-1])
		self.test_data, self.training = trainingTestData(encodedData, 33.0/100.0)
		self.test_data, self.test_labels = getNumpy(self.test_data)
		self.training, self.training_labels = getNumpy(self.training)
		predictions = self.predict(self.test_data, 20)
		print evaluate(predictions, self.test_labels)

	def euclidean_distance(self, row1, row2):
		return np.linalg.norm(row2 - row1)

	def getNeighbors(self, testRow, k):
		distances = []
		for i in range(len(self.training)):
			self.training[i][-1] = self.training_labels[i]
			distances.append((self.training[i], self.euclidean_distance(testRow, self.training[i])))
		distances.sort(key=lambda x:x[1])
		neighbors = []
		for i in range(k):
			neighbors.append(distances[i][0])
		return neighbors

	def getLabel(self, neighbors):
		classVotes = {}
		maxresponse = 0
		actual_response = 0
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		for key, value in classVotes.iteritems():
			if value > maxresponse:
				maxresponse = value
				actual_response = key
		return actual_response

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		return

	def predict(self, features, k):
		predictions = []
		for i in range(len(features)):
			neighbors = self.getNeighbors(features[i], k)
			result = self.getLabel(neighbors)
			predictions.append(result)
		return predictions

class Perceptron:
	def __init__(self):
		#Perceptron state here
		#Feel free to add methods
		return

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		return

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		return

class MLP:
	def __init__(self):
		#Multilayer perceptron state here
		#Feel free to add methods
		return

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		return

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		return

class ID3:
	def __init__(self):
		#Decision tree state here
		#Feel free to add methods
		return

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		return

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		return

if __name__ == "__main__":
	KNN()