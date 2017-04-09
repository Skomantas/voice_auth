'''Machine learning utils.'''

import speech_data as data

import tflearn


def make_model(number_classes):
    # Classification
    tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

    net = tflearn.input_data(shape=[None, 3848])  # Two wave chunks

    net = tflearn.fully_connected(net, 128)
    net = tflearn.dropout(net, 0.5)

    net = tflearn.fully_connected(net, 16)
    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 128)
    net = tflearn.dropout(net, 0.5)

    net = tflearn.fully_connected(net, number_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model


def predict(model, speakers, buffer):
    demo = data.wave_mfcc(buffer)
    result = model.predict([demo])
    conf = numpy.amax(result)*100
    result = data.one_hot_to_item(result, speakers)
    return result, conf
