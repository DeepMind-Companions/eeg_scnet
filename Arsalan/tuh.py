''' The TUH dataset is accessed in tensorflow using a generator object '''
import tensorflow as tf

from .Preprocessing.preprocessing import return_data
from .ReadingFiles.GetFiles import get_files
from joblib import Parallel, delayed

class TUHDataset:
    def __init__(self, datapath):
        self.files = get_files(datapath)

    def generate_individual(self, filename, outputY):
        data = return_data(filename).get_data()
        return data, outputY


    def generate_data(self, batchsize):
        for batchstart in range(0, len(self.files), batchsize):
            filelist = self.files[batchstart:batchstart+batchsize]
            datalist = Parallel(n_jobs=-1)(delayed(self.generate_individual)(file[0], file[1]) for file in filelist)
            for data in datalist:
                yield data

    def create_dataset(self, batchsize):
        return tf.data.Dataset.from_generator(
                self.generate_data,
                args=[batchsize],
                output_signature=(
                    tf.TensorSpec(shape=(21, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.bool)
                )
            )

