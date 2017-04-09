'''Machine learning utils.'''

import tflearn

import speech_data as data


def make_model(number_classes):
    batch = data.wave_batch_generator(batch_size=1000,
                                      target=data.Target.speaker)
    X, Y = next(batch)

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
