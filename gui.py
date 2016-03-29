__author__ = 'jason'
from Tkinter import *
import threading
import os
import Queue
import numpy as np
import itertools
from PIL import Image, ImageTk

from experiment import build_model, get_data
from util import get_filters, numpy_to_im, plot_activations
from keras.callbacks import Callback

class GUIUpdater(Callback):
    def __init__(self, every=20, push_queue=None, pull_queue=None, examples=None, layers=None):
        self.every = every
        self.push_queue = push_queue
        self.pull_queue = pull_queue
        self.layers = layers
        self.examples = dict(enumerate(examples))


    def on_train_begin(self, logs={}):
        self.pull_updates()

    def on_train_end(self, logs={}):
        X, y = data()
        out = self.model.predict(X, verbose=True)
        print(out)

    def pull_updates(self):
        while not self.pull_queue.empty():
            event = self.pull_queue.get()
            if event[0] == 'pause':
                self.model.stop_training = True
            if event[0] == 'delete':
                del self.examples[event[1]]
            if event[0] == 'add':
                self.examples[len(self.examples)] = event[1][np.newaxis, :, :, :]

    def on_epoch_end(self, epoch, logs={}):
        self.pull_updates()
        if epoch % self.every == 0:
            for layer, index, im in get_filters(self.model, 0):
                self.push_queue.put(dict(name='filter', layer=layer, index=index, im=im))

            for j in self.layers:
                for i, example in self.examples.items():
                    for layer, index, im in plot_activations(self.model, j, example):
                        self.push_queue.put(dict(name='activation', layer=layer, index=index, example=i, im=im))


def data():
    X, y = get_data(os.path.join(os.getcwd(), '../data'), junk=True)
    return X, y

def experiment():
    X, y = data()
    model = build_model()
    return model, X, y

def run():
    np.random.seed(1237)
    model, X, y = experiment()
    def train_fun():
        model.fit(X, y, batch_size=len(X), shuffle='batch', show_accuracy=True, nb_epoch=1000, callbacks=[gui.updater])
    gui = GUI(model, X, y, train_fun)
    gui.run()

class GUI(object):
    def __init__(self, model, data, targets, train_fun):
        self.master = Tk()
        self.model = model
        self.data = data
        self.targets = targets
        self.data_view = DataView(self.master, parent=self, training_data=self.data, n_rows=3, n_cols=15)
        self.filter_view = FilterView(self.master, parent=self, nb_filters=len(self.model.get_weights()[0]))
        self.activation_view = ActivationView(len(self.model.get_weights()[0]), [1,2])
        self.train = Button(self.master, text="Train", command=self.train_thread)
        self.train.pack()
        self.data_view.pack()
        self.filter_view.pack()
        self.activation_view.pack()
        self.train_fun = train_fun
        self.pull_queue = Queue.Queue() # A queue for receiving updates from trainer
        self.push_queue = Queue.Queue() # A queue for sending information to trainer

        self.updater = GUIUpdater(every=20, push_queue=self.pull_queue, pull_queue=self.push_queue, examples=[], layers=[1,2])
        self.training = False

    def run(self):
        self.master.mainloop()

    def train_thread(self):
        if not self.training:
            self.train_thread = threading.Thread(target=self.train_fun)
            self.train_thread.start()
            self.master.after(100, self.get_keras_events)
            self.training = True
        else:
            self.push_queue.put(('pause', {}))
            self.training = False

    def add_example(self, data):

        self.push_queue.put(('add', data))
        self.activation_view.add_example(data)

    def get_keras_events(self):
        try:
            while True:
                obj = self.pull_queue.get_nowait()
                if obj['name'] == 'filter':
                    self.filter_view.update_filter(obj['index'], obj['im'])
                if obj['name'] == 'activation':
                    self.activation_view.update_activation(obj['example'], obj['layer'], obj['index'], obj['im'])
#                print(obj)
        except Queue.Empty:
            pass
        finally:
            self.master.after(100, self.get_keras_events)

from functools import partial

class DataView(Frame):
    def __init__(self, *args, **kwargs):
        self.data = kwargs.pop('training_data', None)
        self.rows = kwargs.pop('n_rows', 1)
        self.cols = kwargs.pop('n_cols', 1)
        self.parent = kwargs.pop('parent', None)
        self.images = []
        Frame.__init__(self, *args, **kwargs)
        for i, (row, col) in enumerate(itertools.product(range(self.rows), range(self.cols))):
            if i >= self.data.shape[0]:
                break
            im = self.data[i][0]
            im = numpy_to_im(im)
            im.resize((4*im.width, 4*im.height), Image.NONE)
            im = ImageTk.PhotoImage(im)
            button = Button(self, image=im, command=partial(self.parent.add_example, self.data[i]))
            button.im = im
            button.grid(row=row, column=col)
            self.images.append(button)


class FilterView(Frame):
    def __init__(self, *args, **kwargs):
        self.parent = kwargs.pop('parent', None)
        self.nb_filters = kwargs.pop('nb_filters', 4)
        self.images = []
        Frame.__init__(self, *args, **kwargs)
        for i in range(self.nb_filters):
            button = Button(self)
            button.grid(row=1, column=i)
            self.images.append(button)


    def update_filter(self, n, im):
        im = ImageTk.PhotoImage(im)
        self.images[n].configure(image=im)
        self.images[n].im = im

class ActivationRow(Frame):
    """
    A row of images of activations for a specific example in a specific layer
    """
    def __init__(self, master, im, num_filters, layer, *args, **kwargs):
        Frame.__init__(self, master, *args, **kwargs)
        self.child_elements = {}
        self.num_filters = num_filters
        self.layer = layer
        example_button = Button(self, image=im)
        example_button.im = im
        example_button.grid(row=0, column=0)

        for i in range(num_filters):
            self.child_elements[i] = Button(self)
            self.child_elements[i].grid(row=0, column=i+1)

    def update_image(self, filter_no, im):
        if im.width < 64:
            im = im.resize((64, 64), Image.NONE)
        im = ImageTk.PhotoImage(im)
        self.child_elements[filter_no].configure(image=im)
        self.child_elements[filter_no].im = im

class ActivationView(Frame):
    """
    A frame with many ActivationRows, one for each example/layer pair
    """
    def __init__(self, num_filters, layers, *args, **kwargs):
        self.num_filters = num_filters
        self.layers = layers
        self.examples = []
        self.child_elements = {}
        self.index = 0
        Frame.__init__(self, *args, **kwargs)

    def add_example(self, ex):
        ex = numpy_to_im(ex[0])
        ex = ImageTk.PhotoImage(ex)
        self.child_elements[self.index] = {i : ActivationRow(self, ex, self.num_filters, i, self) for i in self.layers}
        for k, child in self.child_elements[self.index].iteritems():
            child.grid(row=self.index+1, column=k)
        self.index += 1

    def update_activation(self, example, layer, filter_no, im):
        self.child_elements[example][layer].update_image(filter_no, im)


if __name__ == "__main__":
    run()