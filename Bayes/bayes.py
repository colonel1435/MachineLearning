# usr/bin/env python3.4
# Author:Mr.Yuan    fusu1435@163.com
# Date: 2015/07/21
# History: version1.0.0

from numpy import *
import math
import re

# Load data and  label vector & return data list & label vector
def loadDataSet():
	postList = [['my','dog','flea','please','help','problem'],
		['maybe','not','bark','stupid','fuck','beast','him'],
		['my','is','so','cute','love','kind','master'],
		['stop','stupid','garbage','animal','like','how','is'],
		['some','hero','china','pelple','ads','vedio','Q','c++']]
	classVec = [0,1,0,1,0]
	return postList,classVec

# Create a new set of data list & return new list
def createVocabList(dataSet):
#	print(">>> createVocalList\n")
	vocabSet = set([])
	for doc in dataSet:
		vocabSet = vocabSet | set(doc)
	return list(vocabSet)

# set-of-words model : Create vector from words list
def setOfWords2Vec(vocabList,inputSet):
	retVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			retVec[vocabList.index(word)] = 1
		#	print ('the word %s is in List' %word)
		else:
		#	print ('the word %s is not in List' %word)
			pass
	return retVec
	
# bag-of-words model : Create vector from words list
def bagOfWords2Vec(vocabList,inputSet):
#	print(">>> bagOfWords2Vec\n")
	retVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			retVec[vocabList.index(word)] += 1
		else:
			print ('the word %s is not in List/n'%word)
	return retVec

# Train data
def trainData(trainMatrix,trainCategory):
	print(">>> trainData\n")
	numTrainMat = len(trainMatrix)
	numTrainWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainMat)
	p0Num = ones(numTrainWords)
	p1Num = ones(numTrainWords)
	p0Denomi = 0.0
	p1Denomi = 0.0

	for i in range(numTrainMat):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denomi += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denomi += sum(trainMatrix[i])
	
#	p1Vec = p1Num/p1Denomi
#	p0Vec = p0Num/p0Denomi
# Add math log to prevent underflow & wrong result
	p1Vec = log(p1Num/p1Denomi)
	p0Vec = log(p0Num/p0Denomi)
	
	return p0Vec,p1Vec,pAbusive
	
# Bayes classify func
def classifyData(vec2classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2classify*p1Vec)+ log(pClass1)
	p0 = sum(vec2classify*p0Vec)+ log(1.0 - pClass1)
	if p0 > p1:
		return 0
	else:
		return 1


# Test bayes classify func
def testData():
	dataList,clsVec = loadDataSet()
	dataSet = createVocabList(dataList)
	trainMat = []
	for postinDoc in dataList:
		trainMat.append(setOfWords2Vec(dataSet,postinDoc))
	p0Vec,p1Vec,pAb = trainData(array(trainMat),array(clsVec))
	testEntry = ['love','my','animal']
	thisDoc = array(setOfWords2Vec(dataSet,testEntry))
	print (testEntry,'classified as: ',classifyData(thisDoc,p0Vec,p1Vec,pAb))
	
	testEntry = ['stupid','grabage']
	thisDoc = array(setOfWords2Vec(dataSet,testEntry))
	print (testEntry,'classified as: ',classifyData(thisDoc,p0Vec,p1Vec,pAb))

def printList(list):
	for i in range(len(list)):
		print (list[i])

# Parse string to tokens
def parseText(bigString):
	listOfTokens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]

# Test spam & use-email
def testSpam():
	docList = []
	classList = []
	fullList = []
	
	for i in range(1,26):
		wordList = parseText(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullList.extend(wordList)
		classList.append(1)

		bigString = (open('email/ham/%d.txt' % i).read())
		wordList = parseText(bigString)
		docList.append(wordList)
		fullList.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainSet = list(range(50))
	testSet = []
	
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainSet)))
		testSet.append(trainSet[randIndex])
		del(trainSet[randIndex])

	trainMat = []
	trainClass = []
	for docIndex in trainSet:
		trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
		trainClass.append(classList[docIndex])

	p0V,p1V,pSpam = trainData(array(trainMat),array(trainClass))
	errCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
		if classifyData(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
			errCount += 1
	print ('the error rate is:',float(errCount)/len(testSet))


if __name__ == '__main__':
	# dataList,clsVec = loadDataSet()
	# printList(dataList)
	# print (clsVec,'\n')
    #
	# dataSet = createVocabList(dataList)
	# print (dataSet,'\n')
    #
	# inputSet = ('my','stupid','2B','garbage','fuck','R')
	# outputVec = setOfWords2Vec(dataSet,inputSet)
	# print (outputVec,'\n')

	# set-of-words module test
	testData()

	# bag-of-words module test
	#testSpam()
