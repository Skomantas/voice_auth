import audio
import speech_data as data
import ml

from flask import Flask
app = Flask(__name__)


speakers = data.get_speakers()
model = ml.make_model(len(speakers))
model.load('classifier')


@app.route('/auth')
def auth():
    stream = audio.Stream()
    buff = stream.record(1.5)
    sample = audio.stream_to_ints(buff)
    label, conf = ml.predict(model, speakers, sample)
    return label


if __name__ == '__main__':
    app.run()
