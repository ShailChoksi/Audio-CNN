import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
import os
"""from os import listdir
from os.path import isfile, join"""

def oneHotIt(Y):
        m = Y.shape[0]
        Y = Y[:,0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX

def processAudio(bpm,samplingRate,mypath):
        instrIndex = 0
        completeData = []
        completeLabel = []
        testData = []
        testLabel = []
        instruments = []
        testName = ''

        for name in os.listdir(mypath):
                tempPath = os.path.join(mypath, name)
                if os.path.isdir(tempPath) and name.lower() != 'test':
                        instruments.append(name)
                        print(name, ' is ', instrIndex)
                        onlyfiles = [f for f in os.listdir(tempPath) if os.path.isfile(os.path.join(tempPath, f))]

                        for ix,audioFile in enumerate(onlyfiles):
                                audData = scipy.io.wavfile.read(tempPath+'/'+audioFile)
                                seconds = audData[1][:,1].shape[0]/samplingRate
                                samples = (seconds/60) * bpm
                                """print(len(audData[1][:,1][0:math.floor(samples)*((seconds*samplingRate)/samples)]))
                                print((math.floor(samples),(seconds*samplingRate)/samples))"""

                                audData = np.reshape(audData[1][:,1][0:int(int(samples)*((seconds*samplingRate)/samples))],[int(samples),int((seconds*samplingRate)/samples)])
                                for data in audData:
                                        completeData.append(data)
                                completeLabel.append(np.ones([int(samples),1])*instrIndex)
                        instrIndex = instrIndex + 1
                elif os.path.isdir(tempPath) and name.lower() == 'test':
                        testName = tempPath

        if testName != '':
                onlyfiles = [f for f in os.listdir(testName) if os.path.isfile(os.path.join(testName, f))]
                index = -1

                for ix,audioFile in enumerate(onlyfiles):
                        index = -1
                        for instr in instruments:
                                if instr.lower() in audioFile.lower():
                                        index=instruments.index(instr)
                                        break

                        audData = scipy.io.wavfile.read(testName+'/'+audioFile)
                        seconds = audData[1][:,1].shape[0]/samplingRate
                        samples = (seconds/60) * bpm

                        audData = np.reshape(audData[1][:,1][0:int(int(samples)*((seconds*samplingRate)/samples))],[int(samples),int((seconds*samplingRate)/samples)])
                        for data in audData:
                                testData.append(data)
                        testLabel.append(np.ones([int(samples),1])*index)



        Ys = np.concatenate(completeLabel)
        Y = np.concatenate(testLabel)

        specX = np.zeros([len(completeData),1024])
        xindex = 0
        for x in completeData:
                work = matplotlib.mlab.specgram(x)[0]
                worka = work[0:60,:]
                worka = scipy.misc.imresize(worka,[32,32])
                worka = np.reshape(worka,[1,1024])
                specX[xindex,:] = worka
                xindex +=1

        testSpecX = np.zeros([len(testData),1024])
        testXIndex = 0
        for x in testData:
                work = matplotlib.mlab.specgram(x)[0]
                worka = work[0:60,:]
                worka = scipy.misc.imresize(worka,[32,32])
                worka = np.reshape(worka,[1,1024])
                testSpecX[testXIndex,:] = worka
                testXIndex +=1

        split1 = specX.shape[0] - int(specX.shape[0]/20)
        split2 = int((specX.shape[0] - split1) / 2)

        formatToUse = specX
        Data = np.concatenate((formatToUse,Ys),axis=1)
        DataShuffled = np.random.permutation(Data)
        trainX,trainYa = np.hsplit(DataShuffled,[-1])
        """trainX,otherX = np.split(newX,[split1])
        trainYa,otherY = np.split(newY,[split1])
        valX, testX = np.split(otherX,[split2])
        valYa,testYa = np.split(otherY,[split2])
        trainY = oneHotIt(trainYa)
        testY = oneHotIt(testYa)
        valY = oneHotIt(valYa)"""
        trainY = oneHotIt(trainYa)
        testY = oneHotIt(Y)
        #testX = testSpecX
        #testY = testLabel
        return instrIndex,trainX,trainYa,testSpecX,testY

        """split1 = specX.shape[0] - int(specX.shape[0]/20)
        split2 = int((specX.shape[0] - split1) / 2)

        formatToUse = specX
        Data = np.concatenate((formatToUse,Ys),axis=1)
        DataShuffled = np.random.permutation(Data)
        newX,newY = np.hsplit(DataShuffled,[-1])
        trainX,otherX = np.split(newX,[split1])
        trainYa,otherY = np.split(newY,[split1])
        valX, testX = np.split(otherX,[split2])
        valYa,testYa = np.split(otherY,[split2])
        trainY = oneHotIt(trainYa)
        testY = oneHotIt(testYa)
        valY = oneHotIt(valYa)
        return instrIndex,trainX,trainYa,valX,valY,testX,testY"""
