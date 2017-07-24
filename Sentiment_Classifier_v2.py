# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:21:23 2017

@author: Stephen
"""
# Binary Sentiment Classifier for Twitter data
# USAGE: at commandline, enter
# python /path/to/Sentiment_Classifier_v2.py --inputfile /path/to/data --outputfile /desired/path/to/model_file
# This script does the following:
# 1) Shuffles the source file so that modelling is not skewed by ordering in the observations
# 2) Splits the shuffled file into train and test files
# 3) Uses the train file to generate word2vec vectors
# 4) Uses the word2vec vectors obtained in (3) as an embedding layer for a 
#    Convolutional Neural Network
# 5) Saves the convolutional net at a location defined by --outputfile
# 6) Saves the vocabulary mapping to disk
import re
import os
import argparse
import random
import numpy as np
import json
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()



def parseArgs():
    # Get the input arguments and check they're not mangled
    parser = argparse.ArgumentParser('Binary Sentiment Classifier')
    parser.add_argument('--inputfile', help='Fully-qualified path for input file', type=str, required=True)
    parser.add_argument('--outputfile', help='Fully-qualified path to write model to', type=str, default = os.getcwd() + '\\model.h5')
    args = parser.parse_args()
    return(args)
    
    
    
def tokenCleaner(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return ''
 
    
    
def create_embeddings(inputFilename, **params):
    class SentenceGenerator(object):
        def __init__(self, filename):
            self.fname = filename
    
        def __iter__(self):
            for line in open(self.fname, encoding="utf8"):
                line = re.sub('^(.*?,.*?,.*?),','',line)
                yield list(tokenCleaner(line))

    sentences = SentenceGenerator(inputFilename)

    model = Word2Vec(sentences, **params)
    return(model)
     


# Randomise the row order
def shuffleFile(filename):
    with open(filename,'r', encoding="utf8") as source:
        # Skip the header line, we won't need this in any subsequent file
        next(source)
        data = []
        for line in source:
            data.append((random.random(), line))
    data.sort()
    outputFilename = re.sub('\.csv','',filename) + '_shuffled.csv'
    with open(outputFilename,'w', encoding="utf8") as target:
        for _, line in data:
            target.write( line )
    return(outputFilename)



def trainTestSplit(filename, testPercentage=0.2):        
    # Split filename into train and test files with testPercentage of available
    # data in the test file
    
    # This allows us to stream the source file from disk rather than loading 
    # it all
    class FileGenerator(object):
        def __init__(self, filename):
            self.fname = filename
    
        def __iter__(self):
            for line in open(self.fname, encoding="utf8"):
                yield line
            
    # Count how many rows in file by running through them all and keeping
    # a counter
    counter = 0
    sourceFile = FileGenerator(filename)
    for item in sourceFile:
        counter += 1
    
    # Everything from this row onwards goes into the test file instead of the
    # train file
    cutoffRow = round(counter * (1 - testPercentage))
    
    trainFile = re.sub('\.csv','',filename) + '_train.csv'
    testFile = re.sub('\.csv','',filename) + '_test.csv'
    
    # Reset the counter before running through the source file again
    counter = 0
    for item in sourceFile:
        # If we haven't reached the last training row yet, write to the train
        # file
        if counter < cutoffRow:
            with open(trainFile,'a', encoding="utf8") as f:
                f.write(item)
        # If we're past the cutoff row for training, write to the test file
        else:
            with open(testFile,'a', encoding="utf8") as f:
                f.write(item)
        counter += 1
    return([trainFile, testFile])



def generateLabelAndFeatures(filename, vocab):
    labels = []
    features = []

    # This allows us to stream the source file from disk rather than loading 
    # it all
    class FileGenerator(object):
        def __init__(self, filename):
            self.fname = filename
    
        def __iter__(self):
            for line in open(self.fname, encoding="utf8"):
                yield line
    
    # For each line in the file, extract the sentiment label and append it to
    # labels. Then extract the tweet tokens and clean them, before adding them 
    # to features.
    sourceFile = FileGenerator(filename)
    for line in sourceFile:
        # Get the label for this observation and append it to the output list
        tempLabel = int(line.split(',')[1])
        labels.append([tempLabel])
        # Get the tweet, clean and tokenise, then replace each token
        # with the index of the relevant vector in the word2vec model
        # we generated earlier
        line = re.sub('^(.*?,.*?,.*?),','',line)
        line = list(tokenCleaner(line))
        tempFeatures = []
        for item in line:
            if item in vocab:
                tempFeatures.append(vocab[item])
            else:
                tempFeatures.append(0)
        features.append(tempFeatures)
    return([labels, features])



def generateWord2VecMapping(w2vModel, vDim):
    # First, generate a dict that maps vocabulary to the corresponding word2vec
    # vector
    wordVec = {}
    for k,v in w2vModel.wv.vocab.items():
        wordVec[k] = w2vModel.wv.syn0[v.index]

    # Second, generate the dictionary that maps vocabulary to the row in the
    # embedding matrix used in the convolutional net
    vocabDict = dict([(k, v.index + 1) for k, v in w2vModel.wv.vocab.items()])

    # This object tells us how many rows there should be in the embedding matrix
    # One for each row, and one of zeroes for elements that aren't in the 
    # word2vec vocabulary
    n_symbols = len(vocabDict) + 1 # adding 1 to account for 0th index (for masking)    
    
    # Thirdly, combine the first two objects to create the embedding matrix
    embedding_weights = np.zeros((n_symbols,vDim))
    for word,index in vocabDict.items():
        embedding_weights[index,:] = wordVec[word]

    return([embedding_weights, vocabDict])
    
    
    
    
if __name__ == '__main__':
    # Get desired input file and output directory
    arglist = parseArgs()

    # First, make sure that there is no ordering in the source file
    shuffledFile = shuffleFile(arglist.inputfile)
    
    # Split the shuffled source file into train and test files.
    (trainfileLoc, testfileLoc) = trainTestSplit(shuffledFile)
    
    # Parameters for word2vec
    vectorDim = 300 # dimensionality of word vectors
    n_iterations = 10
    n_exposures = 30
    window_size = 5
    
    # Generate word2vec model
    modelOut = create_embeddings(trainfileLoc, 
                                 size = vectorDim,
                                 min_count = n_exposures,
                                 window = window_size,
                                 iter = n_iterations)
    
    # Create two objects:
    # 1) A vocabulary dictionary so that future out-of-sample observations
    #    can be mapped to the existing word2vec model
    # 2) An embedding matrix, with one row for each word2vec mapping and an 
    #    extra row (row zero) containing all zeroes for tweet elements
    #    that are either padding or don't map to the word2vec vocabulary
    (embedding_weights, vocabDict) = generateWord2VecMapping(modelOut, vectorDim)

    # For use in the keras model: this element tells us the input dimension of the
    # word2vec embedding layer
    n_symbols = len(vocabDict) + 1 # adding 1 to account for 0th index (for masking)    
    
    # Load training data and convert tokens to word2vec indices
    (yTrain, xTrain) = generateLabelAndFeatures(trainfileLoc, vocabDict)
    
    # Load test data and convert tokens to word2vec indices
    (yTest, xTest) = generateLabelAndFeatures(testfileLoc, vocabDict)
    
    # Tweet lengths can be uneven, so add null entries such that all tweets
    # have a length equal to MAX_SEQUENCE_LENGTH
    MAX_SEQUENCE_LENGTH = 500
    xTrain = sequence.pad_sequences(xTrain, maxlen=MAX_SEQUENCE_LENGTH)
    xTest = sequence.pad_sequences(xTest, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    # Set up the Convolutional Neural Net that will serve as the classifier
    model = Sequential()
    model.add(Embedding(input_dim=n_symbols, 
                        output_dim=vectorDim,
                        weights=[embedding_weights],
                        input_length=MAX_SEQUENCE_LENGTH))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=2, batch_size=128, verbose=2)

    # Save the model
    model.save(arglist.outputfile)
    
    # Save the vocabulary list (this is needed because we can't evaluate new data without
    # first converting tokens to embedding indices)
    with open(arglist.outputfile + '.vocab', 'w') as f:
        f.write(json.dumps(vocabDict))    