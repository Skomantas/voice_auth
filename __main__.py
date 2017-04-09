import os

import tflearn
import speech_data as data
import tensorflow as tf
import numpy
import pickle
import audio


def test(model,speakers, buffer):
    demo=data.wave_mfcc(buffer)
    result=model.predict([demo])
    conf = numpy.amax(result)*100
    result=data.one_hot_to_item(result,speakers)
    print("predicted : result = %s  confidence = %.2f"%(result,conf))


def make_model(number_classes):
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

    return model


def train(number_classes):
    model = make_model(number_classes)
    model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
    model.save('classifier')
    # with open('classifier.pkl', 'w') as f:
    #   pickle.dump(model,f)


def main():
    speakers = data.get_speakers()
    number_classes=len(speakers)
    print("speakers",speakers)

    model = make_model(number_classes)
    model.load('classifier')

    stream = audio.Stream()
    while True:
      raw_input('press enter to record!!!')
      buff = stream.record(1.5)
      sample = audio.stream_to_ints(buff)
      test(model,speakers,sample)
    # gfx.plot_vector(stream_to_ints(buff))


main()
