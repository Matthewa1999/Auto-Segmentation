import numpy as np
from sklearn import svm
from DataParsing import*
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def training(trainingDirectory):
    counter = 0
    gData = np.array([])
    for entry in os.listdir(trainingDirectory):
        if os.path.isfile(os.path.join(trainingDirectory, entry)) and entry[-4:] == '.npz':

            featOrder, featArr, gTruth, truthWindow = fileOpen(trainingDirectory + '/' + entry)
            if counter == 0:
                featureData = featArr
            else:
                featureData = np.append(featureData, featArr, axis=0)
            gData = np.append(gData, gTruth)
            counter += 1
    clf = svm.SVC(kernel='linear')
    clf.fit(featureData, gData)

    return clf


def testing(testingDirectory, model, smoothing=False):

    pTArr = np.array([])
    pFArr = np.array([])
    rTArr = np.array([])
    rFArr = np.array([])
    fTArr = np.array([])
    fFArr = np.array([])
    aArr = np.array([])

    # plotting and other instantiations
    num = 1
    threshArr = np.arange(0, 20) * 0.5
    pCount = np.zeros(len(threshArr))
    nCount = np.zeros(len(threshArr))
    fPtot, fNtot, tPtot, tNtot = 0, 0, 0, 0
    segmentsArr = np.array([])

    # Print Results
    resultsPrint = True

    # Print segments
    segmentsPrint = True

    # Plot/Save Histograms
    plotPredHist = False
    savePredHist = False

    plotSegHist = False
    saveSegHist = False

    # Print Confusion matrix values
    printConf = False

    # Visualize predictions on gTruth
    viz = False
    saveViz = False
    savePlots = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/VisualizePreds'

    # Ignore Boundaries
    ignoreBoundaries = False

    #Remove extra segments
    segRemove = True



    for entry in os.listdir(testingDirectory):
        if os.path.isfile(os.path.join(testingDirectory, entry)) and entry[-4:] == '.npz':

            featOrder, featArr, gTruth, truthWindow = fileOpen(testingDirectory + '/' + entry)
            preds = model.predict(featArr)

            preds -= 1
            gTruth -= 1

            procPred = postProc(preds, smoothing, segRemove)

            # Plot confusion Matrix

            pT, pF, rT, rF, fT, fF, accuracy, fP, fN, tP, tN, pred = evalAcc(procPred, gTruth, ignoreBoundaries)
            fPtot += fP
            fNtot += fN
            tPtot += tP
            tNtot += tN

            # Visualize Predictions on gTruth
            visualizePred(pred, gTruth, entry, num, viz, saveViz, savePlots)
            num += 1

            samples = len(gTruth)
            pCount, nCount, threshArr = histogramsPredCalc(fP, fN, threshArr, pCount, nCount, samples)


            segmentsArr, numSeg = countSegments(pred, segmentsArr, (segmentsPrint or plotSegHist))

            pTArr = np.append(pTArr, pT)
            pFArr = np.append(pFArr, pF)
            rTArr = np.append(rTArr, rT)
            rFArr = np.append(rFArr, rF)
            fTArr = np.append(fTArr, fT)
            fFArr = np.append(fFArr, fF)
            aArr = np.append(aArr, accuracy)

    results = np.array([round(np.mean(pTArr), 5), round(np.mean(pFArr), 5), round(np.mean(rTArr), 5), round(np.mean(rFArr), 5), round(np.mean(fTArr), 5), round(np.mean(fFArr), 5), round(np.mean(aArr), 5)])

    # Visual stuff
    plotPredHistogram(pCount, nCount, threshArr, smoothing, plotPredHist, savePredHist)
    plotSegHistogram(segmentsArr, smoothing, plotSegHist, saveSegHist)
    confusionMat(fPtot, fNtot, tPtot, tNtot, printConf)
    if segmentsPrint:
        print(np.mean(segmentsArr))
    if resultsPrint:
        print(results)

    return results

def visualizePred(procPred, gTruth, entry, num, plot=False, plotSave = False, saveLoc = ''):
    if plot:
        cArr = []
        sX = []
        sY = []
        x = np.arange(0, len(gTruth)) / 60
        for i in np.arange(0, len(gTruth)):
            if (procPred[i] != gTruth[i]):
                # false negative
                if (procPred[i] == 0):
                    cArr += ['red']
                    sX += [x[i]]
                    sY += [gTruth[i]]
                # false positive
                else:
                    cArr += ['blue']
                    sX += [x[i]]
                    sY += [gTruth[i]]
        # gTruth = np.reshape(gTruth, [1,len(gTruth)])
        # plt.scatter(x, gTruth)
        fig = plt.figure(num, figsize=[12, 7])
        plt.plot(x, gTruth, label=entry[:-4])
        plt.scatter(sX, sY, c=cArr)
        fig.suptitle(entry[:-4])
        plt.xlabel('Time (minutes)\nRed = False Negative; Blue = False Positive')
        plt.ylabel('Classification')
        if plotSave:
            plt.savefig(saveLoc + '/' + entry[:-4] + '.png')

def confusionMat(fP, fN, tP, tN, printCons=False):
    if printCons:
        print("\t\t\t\tPred Nonmusic, Pred Music")
        print("Actual Nonmusic \t" + str(tN) + "\t\t" + str(fP))
        print("Actual Music    \t" + str(fN) + "\t\t" + str(tP))

def countSegments(procPred, segmentsArr, use=False):
    if use:
        diffArr = np.diff(procPred)
        numSeg = (np.sum(np.abs(diffArr))+1)
        segmentsArr = np.append(segmentsArr, numSeg)
    return segmentsArr, numSeg

def plotSegHistogram(segmentsArr, smoothing=False, plot=False, save=False):
    if plot:
        saveLoc = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/Histograms/Segments/'

        # FILE NAME, the rest is automatic
        testGroup = '2018ConcertBandFlute'

        if smoothing:
            smoothStr = ' with smoothing'
            smoothSave = 'WS'
        else:
            smoothStr = ''
            smoothSave = ''

        fig1 = plt.figure(figsize=[12, 6])
        segmentsArr = segmentsArr.astype(int)
        countsArr = np.zeros(int(np.max(segmentsArr)) + 1)
        for ind in segmentsArr:
            countsArr[ind] = countsArr[ind] + 1
        x = np.arange(np.max(segmentsArr) + 1)
        plt.bar(x, countsArr)
        plt.yticks(np.arange(np.max(countsArr) + 2))
        fig1.autofmt_xdate()
        plt.xlabel('Number of Segments')
        plt.ylabel('Number of files')
        plt.suptitle(testGroup + ' Segments' + smoothStr)
        if save:
            plt.savefig(saveLoc + testGroup + smoothSave + '.png')

def histogramsPredCalc(fP, fN, threshArr, pCount, nCount, samples):
    if (fP/samples > np.max(threshArr/100) or fN/samples > np.max(threshArr/100)):
        newPercent = max([np.ceil((fP/samples) * 100), np.ceil((fN/samples) * 100)])
        threshArr = np.arange(0, (newPercent*2 + 1)) * 0.5
        newIndices = len(threshArr)-len(pCount)
        pCount = np.append(pCount, np.zeros(newIndices))
        nCount = np.append(nCount, np.zeros(newIndices))
    pPlace = np.where((fP/samples) < threshArr/100)[0][0]
    pCount[pPlace] = pCount[pPlace] + 1
    nPlace = np.where((fN/samples) < threshArr/100)[0][0]
    nCount[nPlace] = nCount[nPlace] + 1
    return pCount, nCount, threshArr

def plotPredHistogram(pCount, nCount, threshArr, smoothing=False, plot=False, save=False):
    if plot:
        saveLoc = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/Histograms/Predictions/'

        # FILE NAME, the rest is automatic
        testGroup = '2018ConcertBandClar'

        if smoothing:
            smoothStr = ' with smoothing'
            smoothSave = 'WS'
        else:
            smoothStr = ''
            smoothSave = ''

        fig1 = plt.figure(figsize=[12, 4.8])
        x = np.arange(len(pCount))
        plt.bar(x, pCount)
        plt.xticks(x, threshArr)
        fig1.autofmt_xdate()
        plt.xlabel('Percent of fP in samples')
        plt.ylabel('Number of files')
        plt.suptitle(testGroup + ' - false positives' + smoothStr)
        if save:
            plt.savefig(saveLoc + testGroup + 'Fp' + smoothSave + '.png')

        fig2 = plt.figure(figsize=[12, 4.8])
        plt.bar(x, nCount)
        plt.xticks(x, threshArr)
        fig2.autofmt_xdate()
        plt.xlabel('Percent of fN in samples')
        plt.ylabel('Number of files')
        plt.suptitle(testGroup + ' - false negatives' + smoothStr)
        if save:
            plt.savefig(saveLoc + testGroup + 'Fn' + smoothSave + '.png')

def postProc(predictions, smoothing=False, segRemove=False):

    if smoothing:
        predDiff = np.diff(predictions)
        predFixed = np.copy(predictions)

        m1 = np.where(predDiff == -1)[0]
        if np.sum(m1 >= len(predDiff) - 2) > 0:
            m1 = m1[:-np.sum(m1 >= len(predDiff) - 1)]
        m2 = m1[np.where(predDiff[m1 + 1] == 1)[0]] + 1
        predFixed[m2] = 1.0

        m3 = np.where(predDiff == 1)[0]
        if np.sum(m3 >= len(predDiff) - 2) > 0:
            m3 = m3[:-np.sum(m3 >= len(predDiff) - 1)]
        m4 = m3[np.where(predDiff[m3 + 1] == -1)[0]] + 1
        predFixed[m4] = 0.0

        predictions = predFixed

    if segRemove:
        segArray = np.array([])
        segArray, numSeg = countSegments(predictions, segArray, True)
        if numSeg > 11:
            predDiff = np.diff(predictions)
            musToNonMus = np.where(predDiff == -1)[0]
            nonMusToMus = np.where(predDiff == 1)[0]
            if len(musToNonMus) > len(nonMusToMus):
                musToNonMus = musToNonMus[:-1]
            elif len(nonMusToMus) > len(musToNonMus):
                nonMusToMus = nonMusToMus[1:]
            else:
                nonMusToMus = nonMusToMus[1:]
                nonMusToMus = np.append(nonMusToMus, musToNonMus[-1])

            #nonMusToMus = np.insert(nonMusToMus, 0, 0)
            nonMusSections = nonMusToMus - musToNonMus
            while(numSeg > 11 or np.min(nonMusSections[nonMusSections > 0]) < 2):
                shortest = np.argmin(nonMusSections[nonMusSections > 0])
                startInd = musToNonMus[shortest] + 1
                endInd = nonMusToMus[shortest] + 1
                predictions[startInd:endInd] = 1
                nonMusSections[shortest] = np.max(nonMusSections)
                numSeg = numSeg - 2



    return predictions

def evalAcc(predictions, truth, ignoreBoundaries=False):

    truth = np.array(truth.reshape((len(truth), )), dtype=bool)
    predictions = np.array(predictions.reshape((len(predictions), )), dtype=bool)

    if ignoreBoundaries:
        tDiff = np.diff(truth)
        switches = np.where(np.abs(tDiff) == 1)[0]
        predictions[switches + 1] = truth[switches + 1]
        predictions[switches] = truth[switches]

    pT = precision_score(truth, predictions)
    pF = precision_score(~truth, ~predictions)
    rT = recall_score(truth, predictions)
    rF = recall_score(~truth, ~predictions)
    fT = f1_score(truth, predictions)
    fF = f1_score(~truth, ~predictions)

    mask = (predictions == truth)
    accuracyT = np.sum(mask == truth)/np.sum(truth)
    accuracyF = np.sum(np.logical_and(mask, ~truth))/np.sum(~truth)
    accuracy = (accuracyT+accuracyF)/2

    # false positives and negatives for histograms
    fP = np.sum(np.logical_and(~mask, predictions))
    fN = np.sum(np.logical_and(~mask, ~predictions))
    tP = np.sum(np.logical_and(mask, predictions))
    tN = np.sum(np.logical_and(mask, ~predictions))

    return pT, pF, rT, rF, fT, fF, accuracy, fP, fN, tP, tN, predictions
