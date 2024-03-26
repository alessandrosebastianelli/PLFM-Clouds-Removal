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


## Dataset
The dataset will be available soon.

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
