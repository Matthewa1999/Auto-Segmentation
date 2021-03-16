import os
from librosa.core import load
import numpy as np
import scipy as sc
from DataParsing import*
from SVM import*
import pickle


def annotateNewData(audioDirectory, writeAddress, modelPath, fileList=None, generateDataReport=True, keepNPZFiles=True,
            numberOfMusicalExercises=5):
    if fileList is None:
        fileList = []
    _ = writeFeatureData(audioDirectory, '', writeAddress, fileList)

    classifyFeatureData(writeAddress, writeAddress, modelPath, generateDataReport, keepNPZFiles,
                               numberOfMusicalExercises)

# FULL NEW DATA PROCESS ***************************************

audioDirectory = "/Users/matthewarnold/Desktop/MountDirectory/2018-2019/symphonicband"
writeDirectory = "/Users/matthewarnold/Desktop/AutoSeg Local/TextOutput/2018/2018_SB_BariSax"
modelPath = '../Models/2017ABAI.sav'
generateDataReport = True
keepNPZFiles = False
numberOfMusicalExercises = 5

fileList = []

annotateNewData(audioDirectory, writeDirectory, modelPath, fileList, generateDataReport, keepNPZFiles, numberOfMusicalExercises)