
from os.path import dirname
from os.path import join

import numpy as np

data_name = 'data1930.csv'

width = 39
height = 52

n_row = 1930
n_type = 5

class Bunch(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

def load_data(n_class = n_type):

    module_path = dirname(__file__)
    data = np.loadtxt(join(module_path, 'data', data_name),
                      delimiter=',',
                      usecols=(range(0, width*height))
                      )

    target = data[:, 0]
    flat_data = data[:, 1:]

    images = flat_data.view()
    images.shape = (-1, n_row, width*height-1)
    print images.shape


    if n_class < n_type:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    return Bunch(data=flat_data,
                 target=target.astype(np.int),
                 target_names=np.arange(n_type),
                 images=images
                 )