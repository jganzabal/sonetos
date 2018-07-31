import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_sonetos_padded, target_sonetos_padded, NUM_CLASSES, batch_size = 16, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.input_sonetos_padded = input_sonetos_padded
        self.target_sonetos_padded = target_sonetos_padded
        self.shuffle = shuffle
        self.NUM_CLASSES = NUM_CLASSES
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input_sonetos_padded) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X = self.input_sonetos_padded[indexes]
        y = keras.utils.to_categorical(self.target_sonetos_padded[indexes]-1, num_classes=self.NUM_CLASSES)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.input_sonetos_padded))
        print('Epoch ended, reseting indexes')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)