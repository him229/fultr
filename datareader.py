import numpy as np
import os
import pickle as pkl


class DataReader:
    def pickelize_data(self, outpath=None):
        print("Converting the data to pkl files")
        if outpath is not None:
            print("Storing the data in {}".format(outpath))
            pkl.dump(self.data, open(outpath, 'wb'))


def reader_from_pickle(input_filename):
    f = open(input_filename, "rb")
    dr = DataReader()
    dr.data = pkl.load(f)
    return dr
