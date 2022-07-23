from tensorflow.keras.models import load_model
from models.convLSTM import convLSTM
from models.headPLFM import headPLFM
from models.cGAN import cGAN
from models.models_config import *
from utils.datasetHandler import *
from utils.metrics import measure_results
import os

class PLFM:
    def __init__(self, path):
        # Try to load pre-trained LSTM model, if not available initilize it
        try:
            lstm_path = os.path.join(path, 'lstm.h5')
            self.lstm.model = load_model(lstm_path)
        except:
            self.lstm = convLSTM(len_series = LSTM_SETTINGS['SERIES_SIZE'], img_shape = LSTM_SETTINGS['IMAGE_SHAPE'])
        # Try to load pre-trained GAN model, if not available initilize it
        try:
            gan_path = os.path.join(path, 'gan.h5')
            self.gan.generator = load_model(gan_path)
            gan_path = os.path.join(path, 'gan-d.h5')
            self.gan.discriminator = load_model(gan_path)
        except:
            self.gan = cGAN(img_shape = GAN_SETTINGS['IMAGE_SHAPE'])
        # Try to load pre-trained Head of PLFM model, if not available initilize it
        try:
            head_path = os.path.join(path, 'head.h5')
            self.head.model = load_model(head_path)
        except:
            self.head = headPLFM(HEAD_SETTINGS['IMAGE_SHAPE'], self.lstm.model, self.gan.generator)

    def train(self, dataset_path):
        # Train from scratch
        self.lstm = convLSTM(len_series = LSTM_SETTINGS['SERIES_SIZE'], img_shape = LSTM_SETTINGS['IMAGE_SHAPE'])
        self.gan = cGAN(img_shape = GAN_SETTINGS['IMAGE_SHAPE'])

        # Default: Load the proposed dataset
        if dataset_path=='SeriesSen1-2':
            s2_paths, s2_zones = get_images_path(dataset_path, 'sen2')
            s2_images, cloud_masks = split_S2_images(s2_paths)
            s2_series = get_time_series(s2_images)
            s1_images, s1_zones = get_images_path(dataset_path, 'sen1')
            s1_series = get_time_series(s1_images)

        # Print Model Settings
        print('\t @LSTM Settings', LSTM_SETTINGS, '\n')
        self.lstm.train(LSTM_SETTINGS['EPOCHS'], 
               s2_series[:8], # Training
               s2_series[:8], # Validation
               LSTM_SETTINGS['BATCH SIZE'])
        self.lstm.model.save(os.path.join('weights', 'lstm.h5'))

        print('\n\t @GAN Settings', GAN_SETTINGS, '\n')
        self.gan.train(GAN_SETTINGS['EPOCHS'], 
               [s2_series[:8], s1_series[:8]], # Training
               GAN_SETTINGS['BATCH SIZE'])
        self.gan.generator.save(os.path.join('weights', 'gan.h5'))
        self.gan.discriminator.save(os.path.join('weights', 'gan-d.h5'))
        
        print('\n\t @PLFM HEAD Settings', HEAD_SETTINGS, '\n')
        self.head = headPLFM(HEAD_SETTINGS['IMAGE_SHAPE'],  self.lstm.model, self.gan.generator)
        self.head.train(HEAD_SETTINGS['EPOCHS'], 
               [s2_series[:8], s1_series[:8]], # Training
               [s2_series[:8], s1_series[:8]], # Validation
               HEAD_SETTINGS['BATCH SIZE'])
        self.head.model.save(os.path.join('weights', 'head.h5'))

    def test(self, dataset_path):
        pass



