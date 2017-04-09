#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os

import tflearn
import speech_data as data
import tensorflow as tf
import numpy

speakers = data.get_speakers()
number_classes=len(speakers)
print("speakers",speakers)

batch=data.wave_batch_generator(batch_size=1000, target=data.Target.speaker)
X,Y=next(batch)


# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)


net = tflearn.input_data(shape=[None, 3848]) #Two wave chunks

net = tflearn.fully_connected(net, 128)
net = tflearn.dropout(net, 0.5)

net = tflearn.fully_connected(net, 16)
net = tflearn.dropout(net, 0.8)

net = tflearn.fully_connected(net, 128)
net = tflearn.dropout(net, 0.5)

net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)

# demo_file = "8_Vicki_260.wav"
def test(file_name):
  demo_file = file_name
  # demo=data.load_wav_file(data.path + demo_file)
  demo=data.load_wav_file("data/test/" + demo_file)
  result=model.predict([demo])
  conf = numpy.amax(result)*100
  result=data.one_hot_to_item(result,speakers)
  print("predicted speaker for %s : result = %s  confidence = %.2f"%(demo_file,result,conf))





# test("data_FAEM0SA3.WAV")
# test("data_FAEM0SI762.WAV")
# test("data_FALK0SX96.WAV")
# test("data_FCDR1SX376.WAV")
# test("data_skoma1.wav")

 # ~ 97% correct

input('Press ENTER to exit')
