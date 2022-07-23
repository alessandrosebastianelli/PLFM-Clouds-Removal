from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.datasetHandler import image_generatorCycleGAN
import matplotlib.pyplot as plt
import numpy as np
import os

class cGAN():
    def __init__(self, img_shape = (256,256,3)):
        # Input shape
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dataset_name = 'SAR2OPT'

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 128
        self.df = 128

        optimizer = Adam(0.0002)#, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['binary_crossentropy', 'mae'],
                              loss_weights=[1, 80],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

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
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, data_loader, steps, batch_size, sample_interval=200):
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        #s2_series = training_series[0]
        #s1_series = training_series[1]

        #steps = len(s2_series)//batch_size

        #data_loader = image_generatorCycleGAN(s2_series, s1_series, batch_size = batch_size, normalization='minmax', augment=False)
        #----
        
        data_iterator = iter(data_loader)

        for epoch in range(epochs):
            for batch_i in range(steps):
                
                imgs_A, imgs_B = next(data_iterator)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f]" % (epoch, epochs,
                                                                        batch_i, steps,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0]
                                                                        ))

                # If at save interval => save generated image samples
                #if batch_i % sample_interval == 0:
                #    self.sample_images(epoch, batch_i, data_loader)

                if (batch_i%500 == 0) or batch_i==0:

                    fldr = os.path.join('ganlog', str(batch_i))
                    isExist = os.path.exists(fldr)
                    if not isExist:
                        os.makedirs(fldr)

                    self.generator.save(os.path.join(fldr, 'gan.h5'))
                    self.discriminator.save(os.path.join(fldr, 'gan-d.h5'))

                    data_iterator2 = iter(data_loader)
                    s2, s1 = next(data_iterator2)
                    b_preds = self.generator.predict(s1)

                    for b in range(batch_size):
                        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (3*3, 3))
                        axes[0].imshow(0.5*s1[b, ...]+0.5)
                        axes[0].axis(False)
                        axes[0].set_title('Sentinel-1')


                        axes[1].imshow(0.5*s2[b, ...]+0.5)
                        axes[1].axis(False)
                        axes[1].set_title('Sentinel-2')

                        axes[2].imshow(0.5*b_preds[b, ...]+0.5)
                        axes[2].axis(False)
                        axes[2].set_title('Prediction with S1')


                        fig.tight_layout()

                        figpath =  os.path.join(fldr, 'img-{}.png'.format(b))
                        plt.savefig(figpath)
                        plt.close() 
