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
	return (predictions == labels).sum() / float(labels.size)

#===========================================================================================================

class KNN:
	columns = []
	training = []
	test_data = []
	test_labels = []
	training_labels = []

	def __init__(self):
		self.columns = dataset.columns.values
		encodedData = encodeData(dataset,self.columns[5:len(self.columns)-1])
		self.test_data, self.training = trainingTestData(encodedData, 5.0/100.0)
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
	columns = []
	training = []
	test_data = []
	test_labels = []
	training_labels = []
	weights = []

	def prepareData(self):
		self.columns = dataset.columns.values
		self.distance_columns = self.columns[2:]
		normData = normalizeData(dataset, self.columns[2:5])
		encodedData = encodeData(normData,self.columns[5:len(self.columns)-1])
		self.test_data, self.training = trainingTestData(encodedData, 33.0/100.0)
		self.test_data, self.test_labels = getNumpy(self.test_data)
		self.training, self.training_labels = getNumpy(self.training)
		self.weights = np.zeros(len(self.training[0]) + 1)

	def __init__(self):
		self.prepareData()
		self.train(self.training, self.training_labels)
		print evaluate(self.predict(self.test_data), self.test_labels)

	def train(self, features, labels):
		rms = 1.0
		error = 0.0
		learning_rate = 0.01
		l = -1
		while (l<300 and rms != 0.0):
			l+=1
			predictions = []
			index = 0
			for feature in features:
				prediction = self.predictForOne(feature)
				predictions.append(prediction)
				error = labels[index] - prediction
				index +=1
				#update bias
				self.weights[0] = self.weights[0] + learning_rate * error
				for j in range(len(feature)-1):
					self.weights[j+1] = self.weights[j+1] + learning_rate * error * feature[j]
			rms = np.sqrt(((predictions - labels) ** 2).mean())

	def predictForOne(self, feature):
		activation = self.weights[0]
		for i in range(len(feature)-1):
			activation += self.weights[i+1] * feature[i]
		if activation >= 0.0:
			return 1
		return 0

	def predict(self, features):
		predictions = []
		for feature in features:
			predictions.append(self.predictForOne(feature))
		return predictions

class MLP:
	columns = []
	training = []
	test_data = []
	test_labels = []
	training_labels = []
	weightMatrix = []
	transposeMat = []
	weights = []
	nodes = 5
	learning_rate = 0.01

	def initWeightMatrix(self, numberOfNodes, noOfFeatures):
		#placing the bias at the beginning for each node in the hidden layer
		self.weightMatrix = []
		self.weightMatrix.append(np.random.uniform(low=-0.5, high=0.5, size=numberOfNodes))
		for i in range(1,noOfFeatures+1):
			self.weightMatrix.append(np.random.uniform(low=-0.5, high=0.5, size=numberOfNodes))

	def prepareData(self):
		self.columns = dataset.columns.values
		self.distance_columns = self.columns[2:]
		normData = normalizeData(dataset, self.columns[2:5])
		encodedData = encodeData(normData,self.columns[5:len(self.columns)-1])
		self.test_data, self.training = trainingTestData(encodedData, 33.0/100.0)
		self.test_data, self.test_labels = getNumpy(self.test_data)
		self.training, self.training_labels = getNumpy(self.training)
		self.weights = np.random.uniform(low=-0.5, high=0.5, size=self.nodes+1)
		self.initWeightMatrix(self.nodes, len(self.training[0]))

	def hiddenLayerSums(self, feature):
		transposeMat = np.transpose(self.weightMatrix)
		sums = []
		for i in range(0, self.nodes):
			sums.append(self.sum(feature, transposeMat[i]))
		return sums

	def outputLayer(self, inputPred, weights):
		sums = self.sum(inputPred, weights)
		return prediction

	def sum(self, feature, weights):
		g = weights[0]
		for i in range(len(feature)-1):
			g += weights[i+1] * feature[i]
		return g

	def activation(self, g):
		return (1.0/(1+np.exp(-g)))

	def partialDerivative(self, g):
		# act = (1.0/(1+np.exp(-g))) * (1-(1.0/(1+np.exp(-g))))
		return g * (1-g)

	def predict(self, features):
		predictions = []
		for feature in features:
			hiddenLayerSums = self.hiddenLayerSums(feature)
			hiddenActivations = []
			for h in range(0,self.nodes):
				hiddenActivations.append(self.activation(hiddenLayerSums[h]))
			ouputSum = self.sum(hiddenActivations, self.weights)
			outputActivation = self.activation(ouputSum)
			if outputActivation >= 0.5:
				predictions.append(1)
			else:
				predictions.append(0)
		return predictions

	def __init__(self):
		self.prepareData()
		self.train(self.training, self.training_labels)
		print evaluate(self.predict(self.test_data), self.test_labels)

	def train(self, features, labels):
		rms = 1.0
		l = -1
		while (l<100 and rms != 0.0):
			l += 1
			predictions = []
			index = 0
			for feature in features:
				#calculating sums and activation
				hiddenLayerSums = self.hiddenLayerSums(feature)
				hiddenActivations = []
				for h in range(0,self.nodes):
					hiddenActivations.append(self.activation(hiddenLayerSums[h]))
				ouputSum = self.sum(hiddenActivations, self.weights)
				outputActivation = self.activation(ouputSum)

				delta = self.partialDerivative(outputActivation) * (labels[index] - outputActivation)
				hiddenDelta = []
				#backpropagation for the hidden layer
				for i in range(0, self.nodes):
					d = self.partialDerivative(hiddenActivations[i]) * self.weights[i] * delta
					hiddenDelta.append(d)

				#updating weight matrix for hidden layer
				for i in range(0,self.nodes):
					self.weightMatrix[0][i] = self.weightMatrix[0][i] + self.learning_rate * hiddenDelta[i]
					for j in range(len(feature)-1):
						self.weightMatrix[j+1][i] = self.weightMatrix[j+1][i] + self.learning_rate * hiddenDelta[i] * feature[j]

				#updating weights for output layer
				self.weights[0] = self.weights[0] + self.learning_rate * delta
				for j in range(self.nodes-1):
					self.weights[j+1] = self.weights[j+1] + self.learning_rate * delta * hiddenActivations[j]
				index +=1

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
	# print "KNN" , KNN()
	# print "Perceptron", Perceptron()
	print "MLP", MLP()