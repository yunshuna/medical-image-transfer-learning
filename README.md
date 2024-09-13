
# Medical Image Classification with ResNet34 and Transfer Learning

This repository contains code for training and evaluating a ResNet34-based model on medical image datasets using transfer learning techniques. The project explores different methods such as fine-tuning, freezing CNN layers, progressive learning for improving classification performance on medical image datasets. And use cross validation to validate the dataset

## Project Structure

```
├── Resnet
│   ├── batch_predict.py         # Script for batch predictions
│   ├── load_weights.py          # Script to load pre-trained weights
│   ├── model.py                 # Definition of the ResNet34 model and other helper functions
│   ├── predict.py               # Script to run inference on new images
│   ├── train.py                 # Main training script
│   ├── resnet34-pre.pth         # Pre-trained weights for ResNet34(need download by yourself see below link)
│   ├── class_indices.json       # JSON file mapping class indices to labels
│   ├── Fine-tuning.ipynb        # Jupyter notebook for fine-tuning the model
│   ├── Freezing_CNN_Layers.ipynb # Notebook demonstrating layer freezing techniques
│   ├── Progressive_learning.ipynb # Notebook showing progressive learning (unfreezing layers gradually)
│   └── cross_validation.ipynb   # Notebook for 5-fold cross-validation
├── Results                      # Folder to store the results (e.g., plots, metrics)
├── Datasets
│   └── Herlev                   # Example dataset (other datasets should be downloaded separately)
└── README.md                    # Project description and instructions
```

## Usage

download the ResNet34 pre-trained model from    
https://download.pytorch.org/models/resnet34-333f7ec4.pth
and put it on ResNet fold

### 1. Datasets

For this project, we use three medical image datasets:
- BreastMNIST
- Herlev Cervical Cell (example uploaded in `Datasets/Herlev`, but due to quantity restrictions, these are not all the photos, just examples of what the folder should look like)
- ISIC Skin Cancer

The Herlev dataset is provided as an example. You can download the other datasets from the official sources:
- **BreastMNIST**: [https://medmnist.com](https://medmnist.com)
- **ISIC**: [https://isic-archive.com](https://isic-archive.com)
-  **Herlev Cervical Cell**: [http://mde-lab.aegean.gr/images/stories/docs/smear2005.zip](http://mde-lab.aegean.gr/images/stories/docs/smear2005.zip)


Place your datasets in the `Datasets` folder. The folder structure should be as follows:
```
Datasets/
    BreastMNIST/
    Herlev/
    ISIC/
```

### 2. Training the Model

You can start training the ResNet34 model with transfer learning by using the scripts provided. Below are some examples:

- **Fine-tuning**:
  Use the `Fine-tuning.ipynb` notebook to fine-tune layers of the ResNet34 model

- **Freezing CNN Layers**:
  Use the `Freezing_CNN_Layers.ipynb` notebook to freeze specific layers of the ResNet34 model and train the unfrozen layers.

- **Progressive Learning**:
  The `Progressive_learning.ipynb` demonstrates progressive learning where layers are unfrozen gradually during training for fine-tuning.

- **Cross-Validation**:
  Perform 5-fold cross-validation using the `cross_validation.ipynb` notebook to evaluate model performance across different data splits.

### 3. Monitoring Energy Consumption

We use CarbonTracker to monitor the energy consumption during training. This tool tracks the energy consumption and carbon footprint, which is crucial for sustainable AI research, especially in resource-limited environments.

### Results

Results, including accuracy, loss curves, and other metrics, are saved in the `Results` folder. You can customize where results are stored in the respective notebooks or scripts.
