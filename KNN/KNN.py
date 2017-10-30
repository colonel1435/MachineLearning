# -*- coding:utf-8 -*-
#!/usr/bin/python
# Python:          3.4.3
# Platform:        Windows
# Author:          Mr.Yuan	fusu1435@163.com
# Program:         KNN-Algorithm demo ,dating test & handwriting classify
# History:         2015.8.31 v1.0.0

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
import os

def createDataset():
	group = array([[1.0,1.1],[1.0,1.0],[0.0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def KNNclassify(inV,dataSet,labels,kCount):
	datasetSize = dataSet.shape[0]
	diffMat = tile(inV,(datasetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDist = sqDiffMat.sum(axis=1)
	dist = sqDist ** 0.5
	sortedDistIndicies = dist.argsort()
	classCount = {}
	for i in range(kCount):
		voteLabel = labels[sortedDistIndicies[i]]
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1

	sortedClassCout = sorted(classCount.items(),\
				key = operator.itemgetter(1),reverse = True)
	return sortedClassCout[0][0]

def file2matrix(filename):
	fReader = open(filename)
	arrayOfLines = fReader.readlines()
	numOfLines = len(arrayOfLines)
	retMat = zeros((numOfLines,3))
	classLabelVec = []
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		retMat[index,:] = listFromLine[0:3]
		classLabelVec.append(int(listFromLine[-1]))
		index += 1
	return retMat,classLabelVec

def autoNorm(dataSet):
	minVal = dataSet.min(0)
	maxVal = dataSet.max(0)
	ranges = maxVal - minVal
	normDataset = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataset = dataSet - tile(minVal,(m,1))
	normDataset = normDataset/tile(ranges,(m,1))
	
	return normDataset,ranges,minVal

def dateClassTest():
	hRatio = 0.5
	dateMat,dateLabel = file2matrix('datingTest.txt')
	#plotData(dateMat)
	normMat,ranges,minVal = autoNorm(dateMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hRatio)
	errCount = 0.0
	for i in range(numTestVecs):
		classifierRes = KNNclassify(normMat[i,:],normMat[numTestVecs:m,:],\
	dateLabel[numTestVecs:m],3)
		print ("the classifier came back with: %d,the real answer is : %d"\
		% (classifierRes,dateLabel[i]))
	if (classifierRes != dateLabel[i]):
		errCount += 1.0
	print ("the total error rate is : %f" % (errCount/float(numTestVecs)))

def plotData(dataSet):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataSet[:,1],dataSet[:2])
	plt.show()

# conver img to vector
def img2vector(filename):
	retVec = zeros((1,1024))
	fReader = open(filename)
	for i in range(32):
		lineStr = fReader.readline()
		for j in range(32):
			retVec[0,32*i+j] = int(lineStr[j])
	
	return retVec
			
def file2mat(fileDir):
	dataLabels = []
	fileList = os.listdir(fileDir)
	numFile = len(fileList)
	print('numFile = %d' % numFile)
	dataMat = zeros((numFile,1024))
	for i in range(numFile):
		fileNameStr = fileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		dataLabels.append(classNumStr)
		dataMat[i,:] = img2vector('%s/%s' % (fileDir,fileNameStr))
	 
	return dataMat,dataLabels

def loadDataset():
	# Get training data
	print('Get training data:')
	trainingMat,trainingLabels = file2mat('trainingDigits')
	# Get test data
	print('Get test data:')
	testMat,testLabels = file2mat('testDigits')
	
	return trainingMat,trainingLabels,testMat,testLabels
def loadDataset1():
	# Get training data
	trainingLabels = []
	trainingFileList = listdir('trainingDigits')
	numTraining= len(trainingFileList)
	trainingMat = zeros(numTraining,1024)
	for i in range(numTraining):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		trainingLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	
	# Get test data
	testLabels = []
	testFileList = listdir('testDigits')
	numTest = len(testFileList)
	testMat = zeros(numTest,1024)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		testLabels.append(classNumStr)
		testMat[i,:] = img2vector('testDigits/%s' % fileNameStr)

	return trainingMat,trainingLabels,testMat,testLabels
# test hand writing class
def handwritingClassTest():
	# step 1 : load data
	print ('step 1 : load data')
	trainingMat,trainingLabels,testMat,testLabels = loadDataset()
	# step 2 : train data
	print ('step 2 : train data')
	pass
	# step 3 : test data
	errCount = 0.0
	numTest = testMat.shape[0]
	print ('step 3 : test data')
	for i in range(numTest):
		classifierRes = KNNclassify(testMat,trainingMat,trainingLabels,3)
		print ("the classifier cama back with : %d,the real answer is : %d" %(classifierRes,testLabels[i]))
		if (classifierRes != testLabels[i]):
			errCount += 1.0
	print ("\nthe total number of errors is : %d" % errCount)
	print ("\nthe total error rate is : %f" % (errCount/float(numTest)))

if __name__ == '__main__':
#	dataSet,labels = createDataset()
#	print(KNNclassfy([0.4,0.6],dataSet,labels,3))
#	plotData(dataSet)
	dateClassTest()
#	handwritingClassTest()