from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, LeakyReLU, SeparableConv2D
from tensorflow.keras.layers import UpSampling2D, Concatenate, Input, add, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from utils.datasetHandler import image_generatorHEAD

class headPLFM:
    def __init__(self, img_shape, lstm, gan):
        self.img_shape = img_shape
        self.optimizer = Adam(learning_rate = 0.0002)
        self.loss = 'mae'
        self.metrics = ['mse']

        self.gf = 32

        self.model = self.__build()
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)
        self.lstm = lstm
        self.gan = gan


    def __build(self):
        
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False) #256
        d2 = conv2d(d1, self.gf*2) # 128
        d3 = conv2d(d2, self.gf*4) # 64
        d4 = conv2d(d3, self.gf*8) # 32
        d5 = conv2d(d4, self.gf*8) # 16
        #d6 = conv2d(d5, self.gf*8)
        #d7 = conv2d(d6, self.gf*8)

        # Upsampling
        #u1 = deconv2d(d7, d6, self.gf*8)
        #u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(d5, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)
        
        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def __old__build(self):
        inputs = Input(shape=self.img_shape)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = UpSampling2D(2)(x)

            # Project residual
            residual = UpSampling2D(2)(previous_block_activation)
            residual = Conv2D(filters, 1, padding="same")(residual)
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = Conv2D(3, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = Model(inputs, outputs)
        return model
    
    def train(self, epochs, training_series, validation_series, batch_size):
        traingen = image_generatorHEAD(training_series, self.lstm, self.gan, batch_size = batch_size, normalization='minmax', augment=True)
        valgen   = image_generatorHEAD(validation_series, self.lstm, self.gan,  batch_size = batch_size, normalization='minmax', augment=False)

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
