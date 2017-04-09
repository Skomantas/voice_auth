
import speech_data as data
import numpy

import audio
import ml


def test(model,speakers, buffer):
    demo=data.wave_mfcc(buffer)
    result=model.predict([demo])
    conf = numpy.amax(result)*100
    result=data.one_hot_to_item(result,speakers)
    print("predicted : result = %s  confidence = %.2f"%(result,conf))


def train(number_classes):
    model = ml.make_model(number_classes)
    model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
    model.save('classifier')


def main():
    speakers = data.get_speakers()
    number_classes=len(speakers)
    print("speakers",speakers)

    model = ml.make_model(number_classes)
    model.load('classifier')

    stream = audio.Stream()
    while True:
      raw_input('press enter to record!!!')
      buff = stream.record(1.5)
      sample = audio.stream_to_ints(buff)
      test(model,speakers,sample)
    # gfx.plot_vector(stream_to_ints(buff))


main()
