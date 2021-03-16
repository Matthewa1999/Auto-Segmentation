import os
from librosa.core import load
import numpy as np
import scipy as sc
from ProcessInput import*
from Classification_Annotation import*
import pickle


def annotateNewData(audioDirectory, writeAddress, modelPath, fileList=None, generateDataReport=True, keepNPZFiles=True,
            numberOfMusicalExercises=5):

    _ = writeFeatureData(audioDirectory, '', writeAddress, fileList)

    classifyFeatureData(writeAddress, writeAddress, modelPath, generateDataReport, keepNPZFiles,
                               numberOfMusicalExercises)

# FULL NEW DATA PROCESS ***************************************

audioDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/AudioRepo/test"
writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextOutput/test"
modelPath = '../Models/2017ABAI.sav'
generateDataReport = True
keepNPZFiles = False
numberOfMusicalExercises = 5

fileList = None

annotateNewData(audioDirectory, writeDirectory, modelPath, fileList, generateDataReport, keepNPZFiles, numberOfMusicalExercises)