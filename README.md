# Spatio-Temporal SAR-Optical Data Fusion for Cloud Removal via a Deep Hierarchical Model

**Authors: Alessandro Sebastianelli, Erika Puglisi, Maria Pia Del Rosso, Jamila Mifdal, Artur Nowakowki, Fiora Pirri, Pierre Philippe Mathieu and Silvia Liberata Ullo**



The proposed PLFM model combines a time-series of optical images and a SAR image to remove clouds from optical images.


|Cloudy Image|Model Prediction|Ground Truth|
:-----------:|:-----------:|:-----------:
![](res/cloudy.png) | ![](res/prediction.png) | ![](res/gt.png)



## Usage
To train the PLFM you can simply run

```
python main.py --train dataset_path
```

where dataset_path should contain 2 subfolders named "training" and "validation".

To change default parameters please look at [models configuration file](models/models_config.py).


To test the PLFM you can simply run

```
python main.py --test dataset_path
```

where dataset_path is the path to the test dataset.


## Dataset:
The dataset collects roughly 8000 Sentinel-1 (S1) and 8000 Sentinel-2 (S2) images or 2000 S1 and 2000 S2 time-series of 4 images. Each image has a shape of 256x256 pixels; for S1 images only the VV polarisation has been considered; for S2 the red, green and blue bands (including also the QA60 cloud mask band) have been considered.

The dataset is currently shared on google drive:

- [Part 1] https://drive.google.com/drive/folders/1Y8647SFRBS4l5-YK75yz4WyzAx8K4Kou?usp=sharing
- [Part 2] https://drive.google.com/drive/folders/16cF49ZMUn1ROTIxdaH9u74xSs6oqHE2o?usp=sharing
- [Part 3] https://drive.google.com/drive/folders/1Af_V8uY-OAtW4O_L_doSlPsmpueZdd11?usp=sharing



## Cite our papers

The dataset has been created using our tool proposed in: 

    @article{sebastianelli2021automatic,
        title={Automatic dataset builder for Machine Learning applications to satellite imagery},
        author={Sebastianelli, Alessandro and Del Rosso, Maria Pia and Ullo, Silvia Liberata},
        journal={SoftwareX},
        volume={15},
        pages={100739},
        year={2021},
        publisher={Elsevier}
    }


The PLFM is presented in

    @article{sebastianelli2022clouds,
        author={Sebastianelli, Alessandro and Puglisi, Erika and Del Rosso, Maria Pia and Mifdal, Jamila and Nowakowski, Artur and Mathieu, Pierre Philippe and Pirri, Fiora and Ullo, Silvia Libearata},
        title={Spatio-Temporal SAR-Optical Data Fusion for Cloud Removal via a Deep Hierarchical Model},
        journal={Submitted to IEEE Transactions on Geoscience and Remote Sensing},
        publisher={IEEE},
        note = {arXiv preprint arXiv:2106.12226. https://arxiv.org/abs/2106.12226}
    }
