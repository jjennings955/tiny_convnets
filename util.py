from PIL import Image
import numpy as np
from keras.callbacks import Callback
import matplotlib
import matplotlib.pyplot as plt

def get_filters(model, layer=0, type="PIL"):
    filters = model.get_weights()[layer]
    for index, filter in enumerate(filters):
        filter = filter[0]
        if type == "PIL":
            im = numpy_to_im(filter)
            im = im.resize((filter.shape[0]*10, filter.shape[1]*10), Image.NONE)
            yield (layer, index, im)
        if type == "np":
            yield (layer, index, filter)

def plot_filters(model, layer=0, row_size=0, col_size=0, figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    filters = list(get_filters(model, layer, type="np"))
    n_filters = len(filters)
    if n_filters > row_size*col_size:
        raise ValueError()
    for i,(layer, index, im) in enumerate(filters):
        plt.subplot(row_size, col_size, i+1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(im, cmap=plt.cm.gray, interpolation='none')
    return fig


class ReportGenerator(Callback):
    def __init__(self, update_interval, examples):
        self.update_interval = update_interval
        self.examples = examples

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

def numpy_to_im(arr):
    arr -= np.min(arr)
    arr /= np.max(arr)
    im = Image.fromarray(np.uint8(arr*255))
    return im


def plot_activations(model, layer, example):
    for layer, index, activation in get_activations(example, model, layer):
        im = numpy_to_im(activation)
        im = im.resize((activation.shape[0], activation.shape[1]), Image.NONE)
        yield (layer, index, im)

def save_activations(model, layer, epoch, example):
    for layer, index, activation in get_activations(example, model, layer):
        im = numpy_to_im(activation)
        im = im.resize((activation.shape[0], activation.shape[1]), Image.NONE)
        im.save(open('{layer}-{index}-{epoch}-{example_no}.png'))
        yield (layer, index, im)

def get_activations(X, model, layer):
    from keras import backend as K
    get_output = K.function([model.layers[0].input],
                                  [model.layers[layer].get_output(train=False)])
    layer_output = get_output([X])[0]
    for i, activation in enumerate(layer_output[0]):
        yield (layer, i, activation)