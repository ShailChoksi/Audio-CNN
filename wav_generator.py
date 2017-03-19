## Chris - March 17
## Extracts certain beats in a wav file depending on a condition 
## and creates a new wav file with only those beats.
## Currently takes in as an argument a path to the folder
## in which the first file is read/extracted.
## To do: accept an input of neural network output
## (ex: an array of booleans determining whether to include each beat)

import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
import os
from os import listdir
from os.path import isfile, join
import sys
import wave
import struct
arguments = sys.argv
tempPath = str(arguments[1])	# folder
samplingRate = 44100
bpm = 240

f = os.listdir(tempPath)[0]	# just selects first song in folder
print("Exctracting from ",tempPath+'/'+f)
audData = scipy.io.wavfile.read(tempPath+'/'+f)
seconds = audData[1][:,1].shape[0]/samplingRate	# length of file in seconds
samples = int(seconds/60) * bpm	# length of file in beat samples

noise_output = wave.open(f+'_gen_song.wav', 'w')
noise_output.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))

values = []
beat_len = int(samplingRate/bpm*60)	# number of values in one beat

correct_class = True	# condition to include a beat vs silence - to be replaced with classification array

#samples = int(samples/2) # temp to make it shorter for testing - remove later

for i in range(0, samples): # over all beats in song
	if (correct_class):
		correct_class = False
		for j in range(0, beat_len): # over all values in a beat
			value = audData[1][i*beat_len+j,1]
			packed_value = struct.pack('h', value)
			values.append(packed_value)
			values.append(packed_value)
			noise_output.writeframesraw(packed_value)
			noise_output.writeframesraw(packed_value)
	else:
		correct_class = True
		for j in range(0, beat_len): # over all values in a beat
			value = 0
			packed_value = struct.pack('h', value)
			values.append(packed_value)
			values.append(packed_value)
			noise_output.writeframesraw(packed_value)
			noise_output.writeframesraw(packed_value)
	# Progress updates
	if i % (samples/10) == 0:
		print(int(100*i/samples),"%")
		
print("100%")
noise_output.close()
print("output closed")