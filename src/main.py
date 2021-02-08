import os
from librosa.core import load
import numpy as np
import scipy as sc
from DataParsing import*
from SVM import*
import pickle


# FILE WRITER *******************************************
# print("BbClar Missing\n")
# audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/2016-symphonic-Flute"
# textDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextRepo/2016_SB_Flute"
# writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/2016/SB_Flute"
# files = fileWrite(audioDirectory, textDirectory, writeDirectory, True)
#
# print("Flute Missing\n")
# audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/2018_CB"
# textDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextRepo/2018_CB_Flute"
# writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/2018/CB_Flute"
# files = fileWrite(audioDirectory, textDirectory, writeDirectory, True)
#
# print("AltoSax Missing\n")
# audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/2018_CB"
# textDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextRepo/2018_CB_AltoSax"
# writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/2018/CB_AltoSax"
# files = fileWrite(audioDirectory, textDirectory, writeDirectory, True)


# TRAIN AND TEST SVM *******************************************

# modelFile = '/Users/matthewarnold/Desktop/AutoSeg Local/Models/2017ABAI.sav'

# trainingDirectory = '/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/2017'
# clf = training(trainingDirectory)
# pickle.dump(clf, open(modelFile, 'wb'))

modelFile = '/Users/matthewarnold/Desktop/AutoSeg Local/Models/2017ABAI.sav'
model = pickle.load(open(modelFile, 'rb'))
testingDirectory = '/Users/matthewarnold/Desktop/AutoSeg Local/Testing'
results = testing(testingDirectory, model, True)
