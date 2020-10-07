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

    # plotting and other instantiations
    num = 1
    totBinResults = np.empty([7])
    totConfResults = np.array([0, 0, 0, 0])
    totStamps = np.empty([10])
    threshArr = np.arange(0, 20) * 0.5
    pCount = np.zeros(len(threshArr))
    nCount = np.zeros(len(threshArr))
    segmentsArr = np.array([])
    absMeanStamps = np.array([])

    # Print Results
    resultsPrint = True

    # Print segments
    segmentsPrint = True

    # Plot/Save Histograms
    plotPredHist = False
    savePredHist = False
    savePredLoc = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/Histograms/Predictions/'

    plotSegHist = False
    saveSegHist = False
    saveSegLoc = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/Histograms/Segments/'

    plotStampHist = True
    saveStampHist = True
    saveStampLoc = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/Histograms/Stamps/'

    # Print Confusion matrix values
    printConf = False

    # Visualize predictions on gTruth
    viz = False
    saveViz = False
    savePlots = '/Users/matthewarnold/Desktop/AutoSeg Local/Plots/VisualizePreds/RemoveSeg/'

    # Ignore Boundaries
    ignoreBoundaries = False

    # Remove extra segments
    segRemove = True

    # Print Stamp Measure
    stampPrint = True



    for entry in os.listdir(testingDirectory):
        if os.path.isfile(os.path.join(testingDirectory, entry)) and entry[-4:] == '.npz':

            featOrder, featArr, gTruth, truthWindow, blockTimes = fileOpen(testingDirectory + '/' + entry)
            preds = model.predict(featArr)

            preds -= 1
            gTruth -= 1

            procPred = postProc(preds, smoothing, segRemove)

            # Plot confusion Matrix

            binResults, confResults, pred, diffMat = evalAcc(procPred, gTruth, truthWindow, blockTimes, ignoreBoundaries)
            if 10 <= countSegments(pred, segmentsArr, False) <= 11:
                totConfResults += confResults
                totStamps = np.vstack((totStamps, diffMat))
                absMeanStamps = np.append(absMeanStamps, np.mean(np.abs(diffMat)))
                if np.mean(np.abs(diffMat)) > 115:
                    print(entry[:-4])

            # Visualize Predictions on gTruth
            overlayPred(pred, gTruth, entry, num, viz, saveViz, savePlots)
            num += 1

            samples = len(gTruth)
            pCount, nCount, threshArr = histogramsPredCalc(confResults[1], confResults[2], threshArr, pCount, nCount, samples)

            if (segmentsPrint or plotSegHist):
                segmentsArr, numSeg = countSegments(pred, segmentsArr, True)

            totBinResults = np.vstack((totBinResults, binResults))

    totBinResults = totBinResults[1:, :].mean(0)

    # Visuals and Metrics for full file set, rather than individual
    plotPredHistogram(pCount, nCount, threshArr, savePredLoc, smoothing, plotPredHist, savePredHist)
    plotSegHistogram(segmentsArr, saveSegLoc, smoothing, plotSegHist, saveSegHist)
    plotStampHistogram(absMeanStamps, saveStampLoc, smoothing, plotStampHist, saveStampHist)
    confusionMat(totConfResults, printConf)

    totStamps = totStamps[1:, :]
    stampMeanSeg = totStamps.mean(0)
    stampAbsMeanSeg = np.abs(totStamps).mean(0)
    stampStdevSeg = totStamps.std(0)
    stampMean = np.mean(stampMeanSeg)
    stampAbsMean = np.mean(stampAbsMeanSeg)
    stampStdev = np.mean(stampStdevSeg)

    if segmentsPrint:
        print(np.mean(segmentsArr))
    if resultsPrint:
        print(totBinResults)
    if stampPrint:
        with np.printoptions(precision=4, suppress=True):
            print("Mean distance:")
            print(stampMeanSeg)
            print(stampMean)
            print("Abs Mean:")
            print(stampAbsMeanSeg)
            print(stampAbsMean)
            print("Deviation:")
            print(stampStdevSeg)
            print(stampStdev)

    return totBinResults

def overlayPred(procPred, gTruth, entry, num, plot=False, plotSave = False, saveLoc = ''):
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
        fig = plt.figure(num, figsize=[15, 7])
        plt.plot(x, gTruth, label=entry[:-4])
        plt.plot(x, procPred/2)
        # plt.scatter(sX, sY, c=cArr)
        fig.suptitle(entry[:-4])
        plt.xlabel('Time (minutes)\nRed = False Negative; Blue = False Positive')
        plt.ylabel('Classification')
        if plotSave:
            plt.savefig(saveLoc + entry[:-4] + '.png')

def plotPred(pred):
    x = np.arange(0, len(pred)) / 60
    fig = plt.figure(0, figsize=[12, 7])
    plt.plot(x, pred)

def confusionMat(confResults, printCons=False):
    if printCons:
        tN = confResults[0]
        fP = confResults[1]
        fN = confResults[2]
        tP = confResults[3]
        print("\t\t\t\tPred Nonmusic, Pred Music")
        print("Actual Nonmusic \t" + str(tN) + ": " + str(tN/(tN + fP)) + "\t\t" + str(fP)) + ": " + str(fP/(tN + fP))
        print("Actual Music    \t" + str(fN) + ": " + str(fN/(fN + tP)) + "\t\t" + str(tP)) + ": " + str(tP/(fN + tP))

def countSegments(procPred, segmentsArr, use=False):

    diffArr = np.diff(procPred)
    numSeg = (np.sum(np.abs(diffArr))+1)
    if use:
        segmentsArr = np.append(segmentsArr, numSeg)
        return segmentsArr, numSeg
    return numSeg

def plotSegHistogram(segmentsArr, saveSegLoc='', smoothing=False, plot=False, save=False):
    if plot:

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
            plt.savefig(saveSegLoc + testGroup + smoothSave + '.png')

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

def plotPredHistogram(pCount, nCount, threshArr, savePredLoc, smoothing=False, plot=False, save=False):
    if plot:
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
            plt.savefig(savePredLoc + testGroup + 'Fp' + smoothSave + '.png')

        fig2 = plt.figure(figsize=[12, 4.8])
        plt.bar(x, nCount)
        plt.xticks(x, threshArr)
        fig2.autofmt_xdate()
        plt.xlabel('Percent of fN in samples')
        plt.ylabel('Number of files')
        plt.suptitle(testGroup + ' - false negatives' + smoothStr)
        if save:
            plt.savefig(savePredLoc + testGroup + 'Fn' + smoothSave + '.png')

def plotStampHistogram(absMeanStamps, saveLoc='', smoothing=False, plot=False, save=False):
    if plot:

        # FILE NAME, the rest is automatic
        testGroup = '2018ConcertBandBbClar'

        if smoothing:
            smoothStr = ' with smoothing'
            smoothSave = 'WS'
        else:
            smoothStr = ''
            smoothSave = ''

        absMeanStamps = absMeanStamps.astype(int)
        fig1 = plt.figure(figsize=[12, 6])
        absMeanStamps[absMeanStamps < 0] = 0
        countsArr = np.zeros(np.max(absMeanStamps) + 1)
        for ind in absMeanStamps:
            countsArr[ind] = countsArr[ind] + 1
        x = np.arange(np.max(absMeanStamps) + 1)
        plt.bar(x, countsArr)
        plt.yticks(np.arange(np.max(countsArr) + 2))
        fig1.autofmt_xdate()
        plt.xlabel('Abs Mean of Segment Timestamp Differences (s)')
        plt.ylabel('Number of files')
        plt.suptitle(testGroup + ' Stamps' + smoothStr)
        if save:
            plt.savefig(saveLoc + testGroup + smoothSave + '.png')

def postProc(predictions, smoothing=False, segRemove=False):

    if smoothing:
        predDiff = np.diff(predictions)
        predFixed = np.copy(predictions)

        # Fixes 0 1 0
        m3 = np.where(predDiff == 1)[0]
        if np.sum(m3 >= len(predDiff) - 2) > 0:
            m3 = m3[:-np.sum(m3 >= len(predDiff) - 1)]
        m4 = m3[np.where(predDiff[m3 + 1] == -1)[0]] + 1
        predFixed[m4] = 0.0

        # Recalculates diff
        predDiff = np.diff(predFixed)

        # Fixes 1 0 1
        m1 = np.where(predDiff == -1)[0]
        if np.sum(m1 >= len(predDiff) - 2) > 0:
            m1 = m1[:-np.sum(m1 >= len(predDiff) - 1)]
        m2 = m1[np.where(predDiff[m1 + 1] == 1)[0]] + 1
        predFixed[m2] = 1.0

        predictions = predFixed

    if segRemove:
        segArray = np.array([])
        #predictions[0:8] = 0
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

            nonMusSections = nonMusToMus - musToNonMus
            while(numSeg > 11 or np.min(nonMusSections[nonMusSections > 0]) < 2):
                # plotPred(predictions)

                shortest = np.argmin(nonMusSections[nonMusSections > 0])
                startInd = musToNonMus[shortest] + 1
                endInd = nonMusToMus[shortest] + 1
                predictions[startInd:endInd] = 1
                nonMusSections[shortest] = np.max(nonMusSections)
                numSeg = numSeg - 2

            #     plt.clf()
            # plotPred(predictions)
            # plt.clf()


    return predictions

def evalAcc(predictions, truth, stampWindows, blockTimes, ignoreBoundaries=False):

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
    binResults = np.array([pT, pF, rT, rF, fT, fF, accuracy])

    # false positives and negatives for histograms
    fP = np.sum(np.logical_and(~mask, predictions))
    fN = np.sum(np.logical_and(~mask, ~predictions))
    tP = np.sum(np.logical_and(mask, predictions))
    tN = np.sum(np.logical_and(mask, ~predictions))

    confResults = np.array([tN, fP, fN, tP])

    # deviation of time stamps
    predDiff = np.diff(predictions)
    predChange = np.where(abs(predDiff) == 1)[0]
    predStamps = np.empty((5, 2))
    for x in np.arange(0, len(predChange)):
        if x > 10:
            break
        if x % 2 == 0:
            predStamps[int(x/2), 0] = blockTimes[predChange[x] + 1]
        else:
            predStamps[int(x/2), 1] = blockTimes[predChange[x] + 1]
    # if negative, the prediction was early, if positive pred too late
    diffMat = predStamps - stampWindows
    diffMat = np.reshape(diffMat, (1, 10))

    return binResults, confResults, predictions, diffMat
