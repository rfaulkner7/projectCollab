import numpy as np
import math
from numpy.random import rand
from numpy import e
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# You can use the models form sklearn packages to check the performance of your own models


class HateSpeechClassifier(object):
    """Base class for classifiers.
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        """Train your model based on training set

        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass

    def predict(self, X):
        """Predict labels based on your trained model

        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions

        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """

    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this


class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """

    def __init__(self):
        # Add your code here!
        self.hateArr = np.zeros([])
        self.safeArr = np.array([])
        self.total = 0
        self.totalhate = 0
        # raise Exception("Must be implemented")

    def fit(self, X, Y):
        count = 0
        self.hateArr = np.zeros(X[0].size)
        self.safeArr = np.zeros(X[0].size)
        # for y in Y:
        #     print(y)
        for x in X:
            if(Y[count] == 1):
                self.hateArr = np.add(self.hateArr, x)
                self.totalhate += 1
            else:
                self.safeArr = np.add(self.safeArr, x)
            count += 1
        self.total = count
        # for y in self.hateArr:
        #     print(y)
        # self.hateArr = self.hateArr/np.sum(self.hateArr)
        # self.safeArr = self.safeArr/np.sum(self.safeArr)
        # self.hateArr = np.log(self.hateArr/np.sum(self.hateArr))
        # self.safeArr = np.log(self.safeArr/np.sum(self.safeArr))
        self.hateArr = np.log(
            (self.hateArr + 1)/(np.sum(self.hateArr)+self.hateArr.size))
        self.safeArr = np.log(
            (self.safeArr + 1)/(np.sum(self.safeArr)+self.safeArr.size))

        tophate = self.safeArr / self.hateArr
        topsafe = self.hateArr / self.safeArr
        # print(Y[count])
        # if(Y[count] == 0):

        # count = count + 1
        # Add your code here!
        # raise Exception("Must be implemented")

    def predict(self, X):
        # Add your code here!
        # raise Exception("Must be implemented")
        count = 0
        true = 0
        answers = [0]*len(X)
        # print(len(answers))
        # print(answers)
        # return answers
        for x in X:
            safe = np.dot(x, self.safeArr) + \
                math.log((self.total - self.totalhate)/self.total)
            hate = np.dot(x, self.hateArr) + \
                math.log((self.totalhate)/self.total)
            # print(safe >= hate)
            if(safe >= hate):
                # np.append(answers, 0)
                answers[count] = 0
                true += 1
            else:
                # np.append(answers, 1)
                answers[count] = 1
            count += 1
            # print(true)
            # print(count)
        # print(count)
        return answers

# TODO: Implement this


class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        self.wordDict = {}
        self.weights = []
        # raise Exception("Must be implemented")

    def fit(self, X, Y):
        # # Add your code here!
        # epochs = 500
        # rate = .01
        # self.weights = rand(X.shape[1])
        # n = len(X)

        # for i in range(epochs):
        #     scores = np.dot(X, self.weights)
        #     pred = 1 / (1 + np.exp(-scores))

        #     error = Y - pred
        #     calculatedGrad = np.dot(X.T, error)
        #     self.weights += rate * calculatedGrad
        #     # for z in range(self.weights.size):
        #     #     self.weights[z] = calculatedGrad[z] * \
        #     #         rate + np.sum(np.square(self.weights))
        # # raise Exception("Must be implemented")
        m = np.shape(X)[0]
        n = np.shape(X)[1]

        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        self.weights = np.random.randn(n + 1, )

        for i in range(6000):
            y_hat = np.dot(X, self.weights)
            pred = 1/(1+np.exp(-y_hat))

            error = Y - pred

            l2reg = np.sum(np.square(self.weights))

            cost = np.sum(error ** 2) + l2reg

            gradient = (1/m) * (np.dot(X.T, error) + (.1 * self.weights))

            self.weights = self.weights + .1 * gradient

            # print(cost)

    def predict(self, X):
        answers = [0]*len(X)
        count = 0
        for x in X:
            prediction = np.dot(x, self.weights[1:]) + self.weights[0]
            prediction = 1/(1+np.exp(-prediction))
            if(prediction > .5):
                answers[count] = 1
            else:
                answers[count] = 0
            # print(prediction)
            count += 1
        return answers


class PerceptronClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")

    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")

# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayes):


class BonusClassifier(PerceptronClassifier):
    def __init__(self):
        super().__init__()
