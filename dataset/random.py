import numpy as np

def create_dummy_data(dims=[1,2,256,128,16,32,16]):
    return np.random.rand(*dims), np.random.rand(*dims)
