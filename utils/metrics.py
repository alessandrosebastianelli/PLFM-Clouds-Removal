from skimage import metrics
from numpy import cov
from numpy import iscomplexobj
import math
from keras.applications.inception_v3 import InceptionV3
from sewar.full_ref import rmse, sam, uqi
import matplotlib.pyplot as plt
import numpy as np

#Shifted PSNR
def shifted_psnr(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    psnr = metrics.peak_signal_noise_ratio(new_im1, new_im2)
    #print("First PSNR chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            psnr_shift = metrics.peak_signal_noise_ratio(new_im1, new_shifted_image)

            #Choose the biggest psnr
            if psnr_shift > psnr:
                #print("PSNR with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                psnr = psnr_shift

    return psnr

#Shifted SSIM
def shifted_ssim(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    ssim = metrics.structural_similarity(new_im1, new_im2, channel_axis = -1)
    #print("First SSIM chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            ssim_shift = metrics.structural_similarity(new_im1, new_shifted_image, channel_axis = -1)

            #Choose the ssim closest to 1.0
            if ssim_shift > ssim:
                #print("SSIM with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                ssim = ssim_shift

    return ssim

#Frechet Inception Distance(FID) calculation
def calculate_fid(model, act1, act2):
    act1 = act1.reshape(1, act1.shape[0], act1.shape[1], act1.shape[2])
    act2 = act2.reshape(1, act2.shape[0], act2.shape[1], act2.shape[2])
    act1 = model.predict(act1)
    act2 = model.predict(act2)
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = math.sqrt(sigma1.dot(sigma2))

    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + (sigma1 + sigma2 - 2.0 * covmean)

    return fid

#Shifted FID 
def shifted_fid(model, image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    fid = calculate_fid(model, new_im1, new_im2)
    #print("First FID chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            fid_shift = calculate_fid(model, new_im1, new_shifted_image)

            #Choose the lowest fid
            if fid_shift < fid:
                #print("FID with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                fid = fid_shift

    return fid

#Metric developed specifically for this project - CSC-PSNR
def csc_psnr_metric(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]

    #print("First MSE chosen.")
    mse = metrics.mean_squared_error(new_im1, new_im2)

    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            mse_shift = metrics.mean_squared_error(new_im1, new_shifted_image)

            #Choose the lowest mse
            if mse_shift < mse:
                #print("MSE with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                mse = mse_shift

    return mse

#Shifted RMSE
def shifted_rmse(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    rmse1 = rmse(new_im1, new_im2)
    #print("First RMSE chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            rmse_shift = rmse(new_im1, new_shifted_image)

            #Choose the lowest rmse
            if rmse_shift < rmse1:
                #print("RMSE with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                rmse1 = rmse_shift

    return rmse1

#Shifted SAM
def shifted_sam(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    sam1 = sam(new_im1, new_im2)
    #print("First SAM chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            sam_shift = sam(new_im1, new_shifted_image)

            #Choose the lowest sam
            if sam_shift < sam1:
                #print("SAM with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                sam1 = sam_shift

    return sam1

#Shifted UQI
def shifted_uqi(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
       
    uqi1 = uqi(new_im1, new_im2)
    #print("First UQI chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            uqi_shift = uqi(new_im1, new_shifted_image)

            #Choose the uqi closest to 1.0
            if uqi_shift > uqi1:
                #print("UQI with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                uqi1 = uqi_shift

    return uqi1

#Shifted DD(Degree of Distortion)
def DD(ref,tar):
    diff=abs(ref[:]-tar[:]).reshape(-1)
    return np.mean(diff)

def shifted_DD(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    dd = DD(new_im1, new_im2)
    #print("First DD chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            dd_shift = DD(new_im1, new_shifted_image)

            #Choose the lowest dd
            if dd_shift < dd:
                #print("DD with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                dd = dd_shift

    return dd

#Shifted CC(Cross-Correlation)
def CC(ref,tar):
    s=ref.shape #s=(rows,columns,channels)
    tab_cc= np.empty((1,s[2]))
    for idx in range(s[2]):
        R=np.corrcoef(ref[:,:,idx].reshape(-1),tar[:,:,idx].reshape(-1))
        tab_cc[0,idx]=R[0,1]
    
    return np.mean(tab_cc)

def shifted_CC(image1, image2, distance, shift):
    pos_list = (np.arange(1, shift+1).tolist())
    neg_list = [ -x for x in pos_list]
    zero_list = [0]
    x_shift_list = zero_list + pos_list + neg_list
    y_shift_list = zero_list + pos_list + neg_list

    xy_dimension = 256
    xy_dimension_reduction = 256-(distance*2)
    new_im1 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
    new_im2 = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))

    #Image reduction, boundaries are not taken into account
    for i in range(xy_dimension_reduction):
        for j in range(xy_dimension_reduction):
            new_im1[i][j] = image1[i+distance][j+distance]
            new_im2[i][j] = image2[i+distance][j+distance]
 
    cc = CC(new_im1, new_im2)
    #print("First CC chosen.")
 
    #Check of shifted images
    for elemx in x_shift_list:
        for elemy in y_shift_list:
            shifted_image = np.zeros((xy_dimension,xy_dimension,3))
            new_shifted_image = np.zeros((xy_dimension_reduction,xy_dimension_reduction,3))
            for i in range(xy_dimension):
                for j in range(xy_dimension):
                    if ((i+elemx) < 0) or ((i+elemx) > xy_dimension-1):
                        shifted_image[i][j] = 0.0
                    else:
                        if ((j+elemy) < 0) or ((j+elemy) > xy_dimension-1):
                            shifted_image[i][j] = 0.0
                        else:
                            shifted_image[i][j] = image2[i+elemx][j+elemy]
            for i in range(xy_dimension_reduction):
                for j in range(xy_dimension_reduction):
                    new_shifted_image[i][j] = shifted_image[i+distance][j+distance]
            cc_shift = CC(new_im1, new_shifted_image)

            #Choose the cc closest to 1.0
            if cc_shift > cc:
                #print("CC with value x = " + str(elemx) + " and value y = " + str(elemy) + " chosen.")
                cc = cc_shift

    return cc


def measure_results(ground_truth, image_under_test, distance=5, shift=3):
    #PSNR - MAX: INF
    psnr = shifted_psnr(ground_truth, image_under_test, distance, shift)
    #SSIM - MAX: 1.0
    ssim = shifted_ssim(ground_truth, image_under_test, distance, shift)
    #FID - MAX: 0.0
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(246,246,3))
    fid = shifted_fid(model, ground_truth, image_under_test, distance, shift)
    #CSC-PSNR metric - MAX: 0.0
    csc_psnr = csc_psnr_metric(ground_truth, image_under_test, distance, shift)
    #RMSE - MAX:  0.0
    rmse1 = shifted_rmse(ground_truth, image_under_test, distance, shift)
    #SAM - MAX: 0.0
    sam1 = shifted_sam(ground_truth, image_under_test, distance, shift)
    #UQI - MAX: 1.0
    uqi1 = shifted_uqi(ground_truth, image_under_test, distance, shift)
    #DD - MAX: 0.0
    dd = shifted_DD(ground_truth, image_under_test, distance, shift)
    #CC - MAX: 1.0
    cc = shifted_CC(ground_truth, image_under_test, distance, shift)
    return psnr, ssim, fid, csc_psnr, rmse1, sam1, uqi1, dd, cc

    #print('psnr {}\n ssim {}\n fid {}\n csc_psnr {}\n rmse1 {}\n sam1 {}\n uqi1 {}\n dd {}\n cc {}\n'.format(psnr, ssim, fid, csc_psnr, rmse1, sam1, uqi1, dd, cc))