# Generative AI Navigation Information for UAV Reconnaissance in Natural Environments
## Overview
This project focuses on leveraging generative AI to produce realistic images from the perspective of UAVs. The generated images include road and river scenes, which are crucial for efficient and rapid surveying of natural environments.

## Installation
To get started, clone this repository and install the necessary dependencies:
```bash
git clone https://github.com/your-username/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments.git
cd Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments
```
### Using `pip` and `requirements.txt`
```bash
pip install -r requirements.txt
```
### Using `conda` and `environment.yml`
```bash
conda env create -f environment.yml
conda activate aicup
```

## Project Structure
```bash
Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/
├── diffusion/
│   ├── preprocess/
│   ├── pretrained_diffusion/
│   ├── preprocess.py
│   ├── test.py
│   └── train.py
├── gan/
│   ├── data/
│   ├── models/
│   ├── options/
│   ├── util/
│   ├── preprocess.py
│   ├── test.py
│   └── train.py
├── router/
│   └── router.py
├── training_dataset/
│   ├── img/
│   └── label_img/
├── testing_dataset/
│   └── label_img/
├── environment.yml
├── README.md
├── requirements.txt
├── run_diffusion.sh
├── run_gan.sh
└── run_router.sh
```
## Datasets
The `training_dataset` and `testing_dataset` directories contain the datasets provided by the AI CUP. You can replace these datasets with your own data by organizing them in the following structure:
### Training Dataset
* `img/`: Contains raw drone images in .jpg format.
* `label_img/`: Contains black and white images in .png format.

**Note**: The images in `img/` and `label_img/` should have matching filenames (except for the file extensions) and consistent dimensions (428x240 pixels).
### Testing Dataset
* `label_img/`: Contains black and white images in .png format.

## Usage

### Step 1. Diffusion Model

One-click execution to train the model and generate images:
 ```bash
 bash run_diffusion.sh
 ```
The script performs the following steps:
 - `download.py`: Download the pre-trained model.
 - `preprocess.py`: Preprocess the data.
 - `train.py`: Train the model.
 - `test.py`: Generate and test the images.

### Step 2. GAN Model

One-click execution to train the model and generate images:
 ```bash
 bash run_gan.sh
 ```
The script performs the following steps:
 - `preprocess.py`: Preprocess the data.
 - `train.py`: Train the model.
 - `test.py`: Generate and test the images.

### Step 3. Router

Select the final images from both GAN and Diffusion models:
 ```bash
 bash run_router.sh
 ```

## Acknowledgement
We extend our gratitude to the developers of [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [PITI](https://github.com/PITI-Synthesis/PITI) for generously sharing their code, which has been invaluable to our work. Additionally, we would like to thank the developers of [guided-diffusion](https://github.com/openai/guided-diffusion) for providing the pretrained model.
