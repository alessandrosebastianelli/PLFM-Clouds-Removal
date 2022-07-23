from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, MaxPooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from utils.datasetHandler import image_generatorLSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

class convLSTM:
    def __init__(self, len_series=3, img_shape = (256,256,3)):
        self.len_series = len_series
        self.img_shape = img_shape
    
        self.optimizer = Adam(learning_rate=0.0002)
        self.loss = tf.keras.losses.Huber() #huber_loss
        self.metrics = ['mae', 'mse']

        self.model = self.__build()
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)


    def __build(self):
        input_layer = Input(
                shape=(self.len_series, self.img_shape[0], self.img_shape[1], self.img_shape[2])
                )
        # ConvLSTM 1
        x = ConvLSTM2D(filters=128, 
                        kernel_size=(3, 3),
                        activation='relu',
                        padding='same',
                        return_sequences=True)(input_layer)
        x = BatchNormalization()(x)
        # ConvLSTM 2
        x = ConvLSTM2D(filters=64, 
                        kernel_size=(3, 3),
                        activation='relu',
                        padding='same',
                        return_sequences=True)(x)
        x = BatchNormalization()(x)
        # ConvLSTM 3
        x = ConvLSTM2D(filters=32, 
                        kernel_size=(3, 3),
                        activation='relu',
                        padding='same',
                        return_sequences=False)(x)
        # Final Layer
        x = Conv2D(filters=3, kernel_size=(3,3), activation="sigmoid", padding="same")(x)
        # Model
        model = Model(inputs=input_layer, outputs=x)
        return model

    def train(self, epochs, training_series, validation_series, batch_size):
        traingen = image_generatorLSTM(training_series, batch_size = batch_size, normalization='minmax', augment=True)
        valgen   = image_generatorLSTM(validation_series, batch_size = batch_size, normalization='minmax', augment=False)

        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        history = self.model.fit(
            traingen,
            steps_per_epoch = len(training_series)//batch_size,
            validation_data = valgen,
            validation_steps = len(validation_series)//batch_size,
            epochs = epochs,
            callbacks=[es]
        )
        return history
