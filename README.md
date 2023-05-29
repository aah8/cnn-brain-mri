# CNN Brain MRI Classification

This repository contains code and resources for training a Convolutional Neural Network (CNN) to detect tumors brain MRIs. The brain MRIs were obtained via Kaggle https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection.

## Requirements

- Python 3.9
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install the required dependencies using pip:

pip install -r requirements.txt

## Usage

1. Clone the repository:

git clone https://github.com/aah8/cnn-brain-mri.git

2. Install the required dependencies as mentioned in the "Requirements" section.

## Notebooks

1. cnn_base_model

In this notebook, the images are split into a training set (80%) and testing set (20%). A CNN is trained over 30 epochs using an adam (adaptive moment estimation) optimizer and binary crossentropy for the loss function. Predictions are stored in the data/results directory. The following figures are plotted and saved in the figures directory under the base_model subfolder:

A. The training and vaidation set accuracy over epochs
B. The Receiver Operating Characterstic (ROC) curve plotting sensitivity and 1-specificity. 
C. A box plot of the model's predictions by actual ground truth with a dotted line indicating the threshold that optimizes sensitivity and specificity.

2. cnn_model_1

In this notebook,  10-fold cross-validation is used to generate model predictions for all images. For each fold, a CNN is trained over 10 epochs using an adam optimizer, binary crossentropy for the loss function, andd a learning rate of 3e-4. Predictions are stored in the data/results directory. The following figures are plotted and saved in the figures directory under the model_1 subfolder:

A. The Receiver Operating Characterstic (ROC) curve plotting sensitivity and 1-specificity. 
B. A box plot of the models' predictions by actual ground truth with a dotted line indicating the threshold that optimizes sensitivity and specificity.

## Results

There are html versions of the notebooks with all results stored in the notebook_html directory.
