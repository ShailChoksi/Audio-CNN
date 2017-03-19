import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join
from pylab import *

def oneHotIt(Y):
	m = Y.shape[0]
	Y = Y[:,0]
	OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
	OHX = np.array(OHX.todense()).T
	return OHX

def processAudio(bpm,samplingRate,mypath):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	classes = len(onlyfiles)

	dataList = []
	labelList = []
	for ix,audioFile in enumerate(onlyfiles):
		wavData = scipy.io.wavfile.read(mypath+audioFile)

		#wavData[0] = samplingRate, wavData[1] = Data: [leftChannelValues, RightChannelValues]
		rightChannelData = wavData[1][:,1]
		numOfValues = rightChannelData.shape[0]

		seconds = int(numOfValues/samplingRate)
		samples = int((seconds/60)*bpm)
		trimmedData = rightChannelData[0:seconds*samplingRate]
		audData = np.reshape(trimmedData,[samples,int((seconds*samplingRate)/samples)]) #
		#print(audData)
		for data in audData:
			dataList.append(data)
		labelList.append(np.ones([samples,1])*ix)
	Ys = np.concatenate(labelList)

	#print(dataList)
	#print(Ys)
	# dataList = array of size: samples x (seconds*# of songs)/samples in song)]
	# dataList = [[song1p1 # # # .... # # #]
	#			[song1p2 # # # ... # # #]
	#			...
	#			[songNpN # # # ... # # #]]

	specX = np.zeros([len(dataList),1024])
	xindex = 0

	#loop to fill specX with FFT data
	for x in dataList:
		#short time Fourier Transform
		work = matplotlib.mlab.specgram(x)[0]

		#plot(range(0, 129), work)
		#show()
		worka = work[0:60,:]
		worka = scipy.misc.imresize(worka,[32,32])
		worka = np.reshape(worka,[1,1024])
		specX[xindex,:] = worka
		xindex +=1

	split1 = specX.shape[0] - int(specX.shape[0]/20) # 95% of data
	split2 = int((specX.shape[0] - split1) / 2) # 2.5% of data

	formatToUse = specX
	Data = np.concatenate((formatToUse,Ys),axis=1)
	DataShuffled = np.random.permutation(Data)

	#split last column (contains Ys aka song index aka class)
	newX,newY = np.hsplit(DataShuffled,[-1])
	trainX,otherX = np.split(newX,[split1])
	trainYa,otherY = np.split(newY,[split1])
	valX, testX = np.split(otherX,[split2])
	valYa,testYa = np.split(otherY,[split2])
	trainY = oneHotIt(trainYa)
	testY = oneHotIt(testYa)
	valY = oneHotIt(valYa)
	return classes,trainX,trainYa,valX,valY,testX,testY
