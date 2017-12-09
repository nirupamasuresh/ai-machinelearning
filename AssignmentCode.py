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
def preprocess(dataset):
	columns = dataset.columns.values
	encodedData = encodeData(dataset, columns[5:-1])
	normdata = normalizeData(encodedData, columns[2:5])
	return getNumpy(normdata)

def preprocessID3(dataset):
	columns = dataset.columns.values
	normdata = normalizeData(dataset, columns[2:5])
	examples, labels = getNumpy(normdata)
	columns = columns.tolist()
	for a in ["can_id", "can_nam","winner"]:
		columns.remove(a)
	return pd.DataFrame(examples, index=None, columns=columns), labels

class KNN:
	training = []
	training_labels = []

	def __init__(self):
		return

	def euclidean_distance(self, row1, row2):
		return np.linalg.norm(row2 - row1)

	def getNeighbors(self, testRow):
		k = 20
		distances = []
		for i in range(0, len(self.training)):
			labeledTrain = self.training[i].tolist()
			labeledTrain.append(self.training_labels[i])
			distances.append((labeledTrain, self.euclidean_distance(testRow, self.training[i])))
		distances.sort(key=lambda x:x[1])
		neighbors = []
		for i in range(0, k):
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
		self.training = features
		self.training_labels = labels

	def predict(self, features):
		predictions = []
		for i in range(len(features)):
			neighbors = self.getNeighbors(features[i])
			result = self.getLabel(neighbors)
			predictions.append(result)
		return predictions

class Perceptron:
	weights = []
	def __init__(self):
		return

	def train(self, features, labels):
		self.weights = np.random.uniform(-0.1,0.1,len(features[0]) + 1)
		rms = 1.0
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
				for j in range(0, len(feature)):
					self.weights[j+1] = self.weights[j+1] + learning_rate * error * feature[j]
			rms = np.sqrt(((predictions - labels) ** 2).mean())

	def predictForOne(self, feature):
		activation = self.weights[0]
		for i in range(0, len(feature)):
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
	nodes = 5
	weights = []
	weightMatrix = []

	def __init__(self):
		return

	def activation(self, g):
		return (1.0/(1+np.exp(-g)))

	def partialDerivative(self, g):
		return g * (1-g)

	def predict(self, features):
		predictions = []
		for feature in features:
			hiddenActivations, outputActivation = self.computeActivations(feature)
			if outputActivation >= 0.5:
				predictions.append(1)
			else:
				predictions.append(0)
		return predictions

	def computeActivations(self, feature):
		# adding for bias component
		biasFeature = []
		biasFeature.append(1)
		biasFeature.extend(feature)
		# calculating sums and activation
		hiddenLayerSums = np.dot(biasFeature, self.weightMatrix)
		hiddenActivations = []
		for h in range(0, self.nodes):
			hiddenActivations.append(self.activation(hiddenLayerSums[h]))
		# adding bias component
		biasHiddenAct = []
		biasHiddenAct.append(1)
		biasHiddenAct.extend(hiddenActivations)
		ouputSum = np.dot(biasHiddenAct, self.weights)
		outputActivation = self.activation(ouputSum)
		return hiddenActivations, outputActivation

	def train(self, features, labels):
		rms = 1.0
		l = -1
		learning_rate = 0.01
		self.weights = np.random.uniform(low=-0.1, high=0.1, size=self.nodes+1)
		self.weightMatrix = np.random.uniform(-0.1,0.1,(len(features[0])+1, self.nodes))
		while (l<100 and rms != 0.0):
			l += 1
			predictions = []
			index = 0
			for feature in features:
				hiddenActivations, outputActivation = self.computeActivations(feature)
				predictions.append(outputActivation)
				delta = self.partialDerivative(outputActivation) * (labels[index] - outputActivation)
				hiddenDelta = []
				#backpropagation for the hidden layer
				for i in range(0, self.nodes):
					d = self.partialDerivative(hiddenActivations[i]) * self.weights[i] * delta
					hiddenDelta.append(d)

				#updating weight matrix for hidden layer
				for i in range(0,self.nodes):
					self.weightMatrix[0][i] = self.weightMatrix[0][i] + learning_rate * hiddenDelta[i]
					for j in range(0, len(feature)):
						self.weightMatrix[j+1][i] = self.weightMatrix[j+1][i] + learning_rate * hiddenDelta[i] * feature[j]

				#updating weights for output layer
				self.weights[0] = self.weights[0] + learning_rate * delta
				for j in range(self.nodes-1):
					self.weights[j+1] = self.weights[j+1] + learning_rate * delta * hiddenActivations[j]
				index +=1
			rms = np.sqrt(((predictions - labels) ** 2).mean())



class ID3:

	def __init__(self):
		self.attributeData = {}
		self.buckets = [0.2, 0.4, 0.6, 0.8, 1.0]
		self.tree = {}
		self.attributeData['net_ope_exp'] = self.buckets
		self.attributeData['net_con'] = self.buckets
		self.attributeData['tot_loa'] = self.buckets
		self.attributeData['can_off'] = ['H','P','S']
		self.attributeData['can_inc_cha_ope_sea'] = ['INCUMBENT', 'CHALLENGER', 'OPEN']

	def entropy(self, trueValues, total):
		if total == 0.0:
			return 1
		trueValP = trueValues/float(total)
		falseValP = (total-trueValues)/float(total)
		entropy = 0.0
		if trueValP != 0.0:
			entropy -= trueValP * np.log2(trueValP)
		if falseValP != 0.0:
			entropy -= falseValP * np.log2(falseValP)
		return entropy

	def informationGain(self, attribute, examples, labels, presentEntropy):
		gain = 0.0
		selectedAttribute = None
		attributeValues = self.attributeData[attribute]
		valueDict = {}
		index = self.columns.index(attribute)
		for value in attributeValues:
			valueDict[value] = (0,0)
		for i in range(0, len(examples)):
			if examples[i][index] in valueDict.keys():
				positive,total = valueDict[examples[i][index]]
				valueDict[examples[i][index]] = (positive + labels[i], total+1)
			else:
				for b in self.buckets:
					if examples[i][index] <= b:
						if b in valueDict.keys():
							positive,total = valueDict[b]
							valueDict[b] = (positive + labels[i], total+1)
						else:
							valueDict[b] = (labels[i], 1)
						break
		totalEx = len(examples)
		informationGain = presentEntropy

		for value in attributeValues:
			p,t = valueDict[value]
			if t > 0:
				informationGain -= (t/totalEx) * self.entropy(p,t)
		return informationGain

	def mostCommonLabel(self, labels):
		sum = np.sum(labels)
		if sum > (len(labels) - sum):
			return 1
		return 0

	def filterExamples(self, examples, attribute, attributeValue, labels):
		newExamples = []
		newLabels = []
		index = self.columns.index(attribute)
		for i in range(0,len(examples)):
			if examples[i][index] == attributeValue:
				newExamples.append(examples[i])
				newLabels.append(labels[i])
			else:
				for b in self.buckets:
					if examples[i][index] <= b:
						newExamples.append(examples[i])
						newLabels.append(labels[i])
						break
		return newExamples, newLabels

	def decisionTree(self, examples, attributes, parent_examples, labels, parent_labels):
		if len(examples) == 0:
			return self.mostCommonLabel(parent_labels)
		elif self.entropy(np.sum(labels), len(labels)) == 0:
			return labels[0]
		elif len(attributes) == 0:
			return self.mostCommonLabel(labels)
		else:
			infoGain = 0.0
			pickedAttribute = attributes[0]
			for attribute in attributes:
				gain = self.informationGain(attribute, examples, labels, self.entropy(np.sum(labels), len(labels)))
				if gain > infoGain:
					infoGain = gain
					pickedAttribute = attribute
			tree = {pickedAttribute:{}}
			attributes.remove(pickedAttribute)
			for value in self.attributeData[pickedAttribute]:
				subExamples, subLabels = self.filterExamples(examples, pickedAttribute, value, labels)
				subtree = self.decisionTree(subExamples, attributes, examples, subLabels, parent_labels)
				tree[pickedAttribute][value] = subtree
		return tree

	def train(self, features, labels):
		self.columns = features.columns.values.tolist()
		features = features.values
		attributes = []
		attributes.extend(self.columns)
		self.tree = self.decisionTree(features, attributes, [], labels, [])

	def traversal(self, feature, tree):
		if type(tree) is int:
			return tree
		for key in tree.keys():
			if key in self.columns:
				val = feature[self.columns.index(key)]
				if val in tree[key].keys():
					return self.traversal(feature, tree[key][val])
				else:
					for b in self.buckets:
						if val <= b:
							return self.traversal(feature, tree[key][b])

	def predict(self, features):
		predictions = []
		features = features.values
		for feature in features:
			predictions.append(self.traversal(feature, self.tree))
		return predictions

if __name__ == "__main__":
	train_dataset, test_dataset = trainingTestData(dataset, 80.0/100.0)
	train_features, train_labels = preprocess(train_dataset)
	test_features, test_labels = preprocess(test_dataset)

	kNN = KNN()
	kNN.train(train_features, train_labels)
	predictions = kNN.predict(test_features)
	accuracy = evaluate(predictions, test_labels)
	print "knn", accuracy

	perceptron = Perceptron()
	perceptron.train(train_features, train_labels)
	predictions = perceptron.predict(test_features)
	accuracy = evaluate(predictions, test_labels)
	print "perceptron", accuracy

	mlp = MLP()
	mlp.train(train_features, train_labels)
	predictions = mlp.predict(test_features)
	accuracy = evaluate(predictions, test_labels)
	print "mlp", accuracy

	id3 = ID3()
	train_features, train_labels = preprocessID3(train_dataset)
	test_features, test_labels = preprocessID3(test_dataset)
	id3.train(train_features, train_labels)
	predictions = id3.predict(test_features)
	accuracy = evaluate(predictions, test_labels)
	print "id3", accuracy
