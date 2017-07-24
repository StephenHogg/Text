# In this repo, a prototype deep learning classifier for social media sentiment is presented
#
#
# Notes from the script:
#
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
#
# Note: you will need keras, gensim, h5py, tensorflow and dependencies for this script to work
