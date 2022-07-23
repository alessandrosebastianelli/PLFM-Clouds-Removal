import scipy.signal as scisig
import numpy as np
import scipy

def rescale(data, limits):
    return (data - limits[0]) / (limits[1] - limits[0])


def normalized_difference(channel1, channel2):
    num = channel1 - channel2
    den = channel1 + channel2
    den[den == 0] = 0.001  # checking for 0 divisions
    return num / den

def get_cloud_mask(sen2, cloud_threshold=0.2, bin = False):

    #sen2 = sen2 / 10000.
    (ch, r, c) = sen2.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')

    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, rescale(sen2[1], [0.1, 0.5]))
    score = np.minimum(score, rescale(sen2[0], [0.1, 0.3]))
    score = np.minimum(score, rescale((sen2[0] + sen2[10]), [0.4, 0.9]))
    score = np.minimum(score, rescale((sen2[3] + sen2[2] + sen2[1]), [0.2, 0.8]))

    # Clouds are moist
    ndmi = normalized_difference(sen2[7], sen2[11])
    score = np.minimum(score, rescale(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = normalized_difference(sen2[2], sen2[11])
    score = np.minimum(score, rescale(ndsi, [0.8, 0.6]))

    boxsize = 7
    box = np.ones((boxsize, boxsize)) / (boxsize ** 2)

    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if bin:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold]  = 0

    return score