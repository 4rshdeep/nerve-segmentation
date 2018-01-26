# nerve-segmentation

Code for Kaggle competition, Ultrasound Nerve Segmentation more details can be found [here](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data)

### Overview

Architecture used is U-Net: Convolutional Networks for Biomedical Image Segmentation more details [here](arxiv.org/abs/1505.04597)
![Unet](https://github.com/4rshdeep/nerve-segmentation/blob/master/img/u-net-architecture.png)

### Data
Data downloaded from [here](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) is to be kept in ```raw/``` after extracting the train and test zip files.

### Model
Resized images to 96x96 have been fed to the model and used to output masks with values between (0, 1). Output from the network is 96x96 mask which represents a mask that needs to be learned

Loss function used is the negative of Dice Coefficient more details [here](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)

