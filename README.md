# SSDL
Semi-Supervised Dictionary Learning with graph regularized and active points (SSDL-GA)

This is an implementation of paper " SEMI-SUPERVISED DICTIONARY LEARNING WITH GRAPH REGULARIZED AND ACTIVE POINTS " Khanh-Hung TRAN, Fred-Maurice NGOLE MBOULA, Jean-Luc STARCK and Vincent PROST. https://arxiv.org/abs/2009.05964

Require : sklearn, keras, scipy

Objective : We try to build the Dictionary learning model for classification objective that works for both limited number of labelled samples and limited number of unlabelled samples.  

For the MNIST database, we randomly select 200 images from each class, in which 20 images are used for labelled samples, 80 images are used for unlabelled samples and 100 images remain as testing samples.

For the USPS database, we randomly select 110 images from each class, in which 20 images are used for labelled samples, 40 images are used for unlabelled samples and 50 images remain as testing samples.

LP (Label Propagation) is the classical semi supervised method.

Ladder is a semi-supervised neural network approach, implemented by "https://github.com/divamgupta/ladder_network_keras" can achive 98% with only 100 labelled samples and the whole disponible unlabelled samples (for MNIST). 

CNN is a simple convolutional neural network that use only 200 labelled samples for training.

All resting methods are in "Dictionary Learning model " family

| Method \ Data |   USPS       |      MNIST   |
| ------------- | ------------- | -------------|
|LP| 90.3 ± 1.3 | 85.12 ± 0.6 |
| ------------- | ------------- | -------------|
|OSSDL | 80.8 ± 2.8 | 73.2 ± 1.8 |
|SD2D | 86.6 ± 1.6 | 77.6 ± 0.8 |
|SSR-D | 87.2 ± 0.5 | 83.8 ± 1.2 |
|SSP-DL | 87.8 ± 1.1 | 85.8 ± 1.2 |
|USSDL | 91.56 ± 1.15 | 84.8 ± 1.7 |
|PSSDL* | 86.9 ± 1.0 | 87.4 ± 1.2 |
|SSD-LP | 90.3 ± 1.3 | 87.8 ± 1.6 |
| SSDL-GA       | 93.6 ± 1.0  |    90 ± 0.8      |
| ------------- | ------------- | -------------|
| CNN           | 89.28 ± 1.4  |     88.4 ± 1.1|
| Ladder        |  92.68 ± 1     |    89.84  ± 0.8      |

(*) In PSSDL, for each class, we use 25 images as labelled samples instead of 20, the rest of the training data as unlabelled samples and all testing data for testing samples.
