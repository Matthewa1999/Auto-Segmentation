import os
from librosa.core import load
import numpy as np
import scipy as sc
from DataParsing import*
from SVM import*
import pickle


def newData(audioDirectory, writeAddress, modelPath, generateDataReport=True, keepNPZFiles=True,
            numberOfMusicalExercises=5):
    _ = fileWrite(audioDirectory, writeAddress)

    newDataClassificationWrite(writeAddress, writeAddress, modelPath, generateDataReport, keepNPZFiles,
                               numberOfMusicalExercises)

# FILE WRITER *******************************************
# print("BbClar Missing\n")
# audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/2018-symphonic-Alto Saxophone"
# textDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextRepo/2016-middle-Flue-ann"
# writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/NewDataNPZ/2018/SB-AltoSax"
# files = fileWrite(audioDirectory, textDirectory, writeDirectory, True, True)


# audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/2018-symphonic-Alto Saxophone"
# textDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextRepo/2016-middle-Flue-ann"
# writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/NewDataNPZ/2018/SB-AltoSax"
# files = fileWrite(audioDirectory, "", writeDirectory, True)


# TRAIN AND TEST SVM *******************************************

# modelFile = '/Users/matthewarnold/Desktop/AutoSeg Local/Models/2017ABAI.sav'

# trainingDirectory = '/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/2017'
# clf = training(trainingDirectory)
# pickle.dump(clf, open(modelFile, 'wb'))

# modelFile = '/Users/matthewarnold/Desktop/AutoSeg Local/Models/2017ABAI.sav'
# model = pickle.load(open(modelFile, 'rb'))
# testingDirectory = '/Users/matthewarnold/Desktop/AutoSeg Local/Testing'
# results = testing(testingDirectory, model, True)




# FULL NEW DATA PROCESS ***************************************


audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/2018-symphonic-Alto Saxophone"
npzDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/NewDataNPZ/2018/test"
writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/NPZ Repo/NewDataNPZ/2018/test"
modelPath = '/Users/matthewarnold/Desktop/AutoSeg Local/Models/2017ABAI.sav'
generateDataReport = True
keepNPZFiles = False
numberOfMusicalExercises = 5

# newData(audioDirectory, writeDirectory, modelPath, generateDataReport, keepNPZFiles, numberOfMusicalExercises)
newDataClassificationWrite(writeDirectory, writeDirectory, modelPath, generateDataReport, keepNPZFiles,
                               numberOfMusicalExercises)