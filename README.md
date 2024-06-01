# Generative AI Navigation Information for UAV Reconnaissance in Natural Environments

## Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Project Structure](#Project-Structure)
- [Datasets](#Datasets)
- [Reproduction](#Reproduction)
- [Usage](#Usage)
- [Acknowledgements](#Acknowledgement)
- [Arguments](#Arguments)
- [Results](#Results)

## Overview
> Obtaining real-world images from the perspective of UAVs can be costly. Generative AI, on the other hand, can produce a substantial amount of realistic data with a limited dataset. Therefore, this project will utilize generative AI to generate images of roads and rivers from the viewpoint of UAVs under specified conditions.

We employ two models: GAN ([pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)) and Diffusion([PITI](https://github.com/PITI-Synthesis/PITI)). The raw data is fed into both models. The Diffusion model utilizes an [guided-diffusion](https://github.com/openai/guided-diffusion) pre-trained model for fine-tuning, while the GAN model is trained from scratch. The generated images are evaluated by a Router, which determines the final output by selecting the best result from either the GAN or Diffusion model.

![architecture](https://github.com/Shengwei0516/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/imgs/architecture.png)

## Installation
To get started, clone this repository and install the necessary dependencies:
```bash
git clone https://github.com/your-username/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments.git
cd Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments
```
* Using `pip` and `requirements.txt`
```bash
pip install -r requirements.txt
```
* Using `conda` and `environment.yml`
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
├── run_reproduce.sh
└── run_router.sh
```

## Datasets
The `training_dataset` and `testing_dataset` directories contain the datasets provided by the [AI CUP 2024](https://tbrain.trendmicro.com.tw/Competitions/Details/34). You can replace these datasets with your own data by organizing them in the following structure:
* Training Dataset
    * `img/`: Contains raw drone images in .jpg format.
![img](https://github.com/Shengwei0516/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/training_dataset/img/TRA_RI_1000000.jpg)
    * `label_img/`: Contains black and white images in .png format.
![label_img](https://github.com/Shengwei0516/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments/blob/main/training_dataset/label_img/TRA_RI_1000000.png)
* Testing Dataset
    * `label_img/`: Contains black and white images in .png format.

**Note**: The images in `img/` and `label_img/` should have matching filenames (except for the file extensions) and consistent dimensions. Filenames for road data should include **RO** and filenames for river data should include **RI**.

## Reproduction
One-click execution to reproduce the best results:
```bash
bash run_reproduce.sh
 ```
The script performs the following steps:
 - Uses `gdown` to download the best checkpoints to `./checkpoints`
 - `gan/preprocess.py`: Preprocess the data for GAN.
 - `gan/test.py`: Use GAN generate images.
 - `diffusion/preprocess.py`: Preprocess the data for Diffusion.
 - `diffusion/test.py`: Use Diffusion generate images.
 - `router/router.py`: Use Router selects the final best results.

## Usage
**Warning**: Executing the scripts below requires approximately **32GB** of VRAM. If your hardware does not meet this requirement, you may need to adjust the [Arguments](#Arguments) accordingly.

### Step 1. Diffusion
One-click execution to train the model and generate images:
 ```bash
 bash run_diffusion.sh
 ```
The script performs the following steps:
 - `download.py`: Download the pre-trained model.
 - `preprocess.py`: Preprocess the data.
 - `train.py`: Train the model.
 - `test.py`: Generate images.

### Step 2. GAN
One-click execution to train the model and generate images:
 ```bash
 bash run_gan.sh
 ```
The script performs the following steps:
 - `preprocess.py`: Preprocess the data.
 - `train.py`: Train the model.
 - `test.py`: Generate images.

### Step 3. Router
Select the final images from both GAN and Diffusion models:
 ```bash
 bash run_router.sh
 ```
The script performs the following steps:
 - `router.py`: Selects the final results.

## Acknowledgement
We extend our gratitude to the developers of [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [PITI](https://github.com/PITI-Synthesis/PITI) for generously sharing their code, which has been invaluable to our work. Additionally, we would like to thank the developers of [guided-diffusion](https://github.com/openai/guided-diffusion) for providing the pretrained model.

We also thank [AI CUP 2024](https://tbrain.trendmicro.com.tw/Competitions/Details/34) for organizing the competition and providing the datasets.

## Arguments
The scripts `train.py` and `test.py` in the diffusion and gan directory share various configurable arguments. Below are the explanations for some of the key arguments:

### `diffusion/`

| Argument              | Description                                                                                                                              | Default Value |
|:--------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| **data_dir**          | The directory where the training data is stored. This should be a path to the folder containing the training images.                     | ""            |
| **val_data_dir**      | The directory where the validation data is stored. This is used to evaluate the model during training.                                   | ""            |
| **model_path**        | The path where the trained model will be saved. This allows you to specify where to store the model checkpoints.                         | ""            |
| **encoder_path**      | The path to the pre-trained encoder model. This is used if the training process requires a pre-trained encoder.                          | ""            |
| **schedule_sampler**  | The method for sampling the training data. Default is "uniform", which samples data uniformly.                                           | "uniform"     |
| **lr**                | The learning rate for the optimizer. Controls the step size at each iteration while moving toward a minimum of the loss function.        | 1e-4          |
| **weight_decay**      | The weight decay (L2 penalty) for the optimizer. Helps to prevent overfitting by penalizing large weights.                               | 0.0           |
| **lr_anneal_steps**   | The number of steps over which the learning rate is annealed. This helps in gradually reducing the learning rate as training progresses. | 0             |
| **batch_size**        | The number of samples processed before the model is updated.                                                                             | 1             |
| **microbatch**        | The size of microbatches. -1 disables microbatches.                                                                                      | -1            |
| **ema_rate**          | The rate for exponential moving average (EMA) of model parameters. Helps to smooth out the training process.                             | 0.9999        |
| **log_interval**      | The number of iterations between logging the training status.                                                                            | 200           |
| **save_interval**     | The number of iterations between saving the model checkpoint.                                                                            | 20000         |
| **resume_checkpoint** | The path to a checkpoint file to resume training from a previous state. Allows you to continue training from where it left off.          | ""            |
| **use_fp16**          | Boolean indicating whether to use 16-bit floating-point precision. Can reduce memory usage and speed up training on compatible hardware. | False         |
| **fp16_scale_growth** | The growth factor for the loss scaling used in 16-bit precision training.                                                                | 1e-3          |
| **super_res**         | An integer flag to indicate if super-resolution is to be used.                                                                           | 0             |
| **sample_c**          | A parameter controlling the sampling process.                                                                                            | 1.0           |
| **sample_respacing**  | The respacing strategy for sampling.                                                                                                     | 100           |
| **uncond_p**          | The probability of using an unconditional model during training.                                                                         | 0.2           |
| **num_samples**       | The number of samples to generate.                                                                                                       | 1             |
| **finetune_decoder**  | Boolean indicating whether to fine-tune the decoder. Allows for further training of the decoder part of the model.                       | False         |
| **mode**              | A parameter to specify the mode of operation, such as training, evaluation, etc.                                                         | ""            |
### `gan/`

| Argument              | Description                                                                                           | Default Value               |
|-----------------------|-------------------------------------------------------------------------------------------------------|-----------------------------|
| **dataroot**          | Path to images (should have subfolders trainA, trainB, valA, valB, etc).                               | Required                    |
| **name**              | Name of the experiment. It decides where to store samples and models.                                  | 'experiment_name'           |
| **gpu_ids**           | GPU ids: e.g., '0', '0,1,2', '0,2'. Use -1 for CPU.                                                   | '0'                         |
| **checkpoints_dir**   | Directory where models are saved.                                                                     | './checkpoints'             |
| **seed**              | Random seed for reproducibility.                                                                      | 0                           |
| **model**             | Chooses which model to use. [cycle_gan | pix2pix | test | colorization].                             | 'cycle_gan'                 |
| **input_nc**          | Number of input image channels: 3 for RGB and 1 for grayscale.                                        | 3                           |
| **output_nc**         | Number of output image channels: 3 for RGB and 1 for grayscale.                                       | 3                           |
| **ngf**               | Number of generator filters in the last convolution layer.                                            | 64                          |
| **ndf**               | Number of discriminator filters in the first convolution layer.                                       | 64                          |
| **netD**              | Discriminator architecture [basic | n_layers | pixel].                                                | 'basic'                     |
| **netG**              | Generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128].                      | 'resnet_9blocks'            |
| **n_layers_D**        | Number of layers in the discriminator if netD is 'n_layers'.                                          | 3                           |
| **norm**              | Normalization type [instance | batch | none].                                                         | 'instance'                  |
| **init_type**         | Network initialization method [normal | xavier | kaiming | orthogonal].                               | 'normal'                    |
| **init_gain**         | Scaling factor for normal, xavier, and orthogonal initialization.                                     | 0.02                        |
| **no_dropout**        | If specified, do not use dropout for the generator.                                                   | Action (store_true)         |
| **dataset_mode**      | Chooses how datasets are loaded [unaligned | aligned | single | colorization].                        | 'unaligned'                 |
| **direction**         | Direction of the transformation [AtoB | BtoA].                                                        | 'AtoB'                      |
| **serial_batches**    | If true, takes images in order to make batches, otherwise takes them randomly.                        | Action (store_true)         |
| **num_threads**       | Number of threads for loading data.                                                                   | 4                           |
| **batch_size**        | Input batch size.                                                                                     | 1                           |
| **load_size**         | Scale images to this size.                                                                            | 286                         |
| **crop_size**         | Crop images to this size.                                                                             | 256                         |
| **max_dataset_size**  | Maximum number of samples allowed per dataset. If the dataset directory contains more, only a subset is loaded. | float("inf")                |
| **preprocess**        | Image preprocessing method [resize_and_crop | crop | scale_width | scale_width_and_crop | none].       | 'resize_and_crop'           |
| **no_flip**           | If specified, do not flip the images for data augmentation.                                           | Action (store_true)         |
| **display_winsize**   | Display window size for both visdom and HTML.                                                         | 256                         |
| **n_epochs**          | Number of epochs with the initial learning rate.                                                      | 100                         |
| **n_epochs_decay**    | Number of epochs to linearly decay the learning rate to zero.                                         | 100                         |
| **beta1**             | Momentum term of adam optimizer.                                                                      | 0.5                         |
| **lr**                | Initial learning rate for adam optimizer.                                                             | 0.0002                      |
| **gan_mode**          | Type of GAN objective [vanilla | lsgan | wgangp].                                                     | 'lsgan'                     |
| **pool_size**         | Size of image buffer that stores previously generated images.                                         | 50                          |
| **lr_policy**         | Learning rate policy [linear | step | plateau | cosine].                                              | 'linear'                    |
| **lr_decay_iters**    | Number of iterations after which learning rate is multiplied by a gamma.                              | 50                          |

These arguments offer flexibility in training and testing the diffusion model, allowing you to fine-tune the process according to your specific requirements and hardware capabilities.

## Results
Below is a table showcasing the results of image generation for different environments:
|   ID   |  Method   |    Generator    |   Resize    |  Data Type   | Batch Size |  LR  |  Epochs  |   PUB FID    |   PRI FID    |       Note        |
|:------:|:---------:|:---------------:|:-----------:|:------------:|:----------:|:----:|:--------:|:------------:|:------------:|:-----------------:|
|   01   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   200    |   149.5899   |      -       |                   |
|   02   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    256     | 2e-4 |   200    |   163.8351   |      -       |                   |
|   03   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   400    |   133.0080   |      -       |                   |
|   04   |    GAN    | ResNet 9blocks  |  Bilinear   |    Mixed     |     64     | 2e-4 |   200    |   267.8923   |      -       |                   |
|   05   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   400    |   133.7452   |      -       |       Extra       |
|   06   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   600    |   129.6689   |      -       |                   |
|   07   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   1000   |   134.3076   |      -       |                   |
|   08   |    GAN    |      U-Net      |  Bilinear   |    Mixed     |     16     | 2e-4 |   200    |   137.9879   |      -       |                   |
|   09   |    GAN    |    U-Net 512    |  Bilinear   |    Mixed     |     16     | 2e-4 |   200    |   141.4001   |      -       |                   |
|   10   |    GAN    |      U-Net      |  Bilinear   |    Mixed     |     64     | 2e-4 |   600    |   142.0793   |      -       |                   |
|   11   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |     64     | 2e-4 |   200    |   135.5488   |      -       |                   |
|   12   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   400    |   127.8701   |      -       |                   |
|   13   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   600    |   156.7936   |      -       |                   |
|   14   |    GAN    |       VAE       |  Bilinear   |    Mixed     |     8      | 2e-4 |   200    |   133.0856   |      -       |                   |
|   15   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     1      | 2e-4 |   200    |   206.0832   |      -       |                   |
|   16   | Diffusion | SDv1-5 Scribble |  Bilinear   |    Mixed     |     8      | 1e-5 | 1e4 step |   186.8134   |      -       |                   |
|   17   | Diffusion | SDv1-5 Softedge |  Bilinear   |    Mixed     |     8      | 1e-5 | 1e4 step |   211.2076   |      -       |                   |
|   18   |    GAN    |    U-Net 256    |  Bilinear   |    12 out    |     64     | 2e-4 |   400    |   127.1882   |      -       |     Synthetic     |
|   19   |    GAN    |    U-Net 256    |  Bilinear   |    18 out    |     64     | 2e-4 |   400    |   123.7632   |      -       |     Synthetic     |
|   20   |    GAN    |    U-Net 256    |  Bilinear   |    19 out    |     64     | 2e-4 |   400    |   124.1638   |      -       |     Synthetic     |
|   21   |    GAN    |    U-Net 512    |  Bilinear   |   Separate   |     16     | 2e-4 |   400    |   136.8711   |      -       |                   |
|   22   | Diffusion | SDv1-5 Scribble |  Bilinear   |    Mixed     |     64     | 1e-5 | 1e4 step |   190.1620   |      -       |                   |
|   23   |   Mixed   |     19 + 22     |  Bilinear   |   Separate   |     -      |  -   |    -     |   144.5344   |      -       |                   |
|   24   | Diffusion |       22        |  Bilinear   |    Mixed     |     -      |  -   |    -     |   192.3814   |      -       |  FP32 inference   |
|   25   | Diffusion | SDv1-5 Scribble |  Bilinear   |    Mixed     |     64     | 1e-5 | 2e4 step |   200.9103   |      -       |                   |
|   26   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    256     | 2e-4 |   400    |   130.6916   |      -       |                   |
|   27   |    GAN    |     FSRCNN      |  Bilinear   |    19 out    |     -      |  -   |    -     |   153.7523   |      -       |  Denoise filter   |
|   28   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    256     | 2e-4 |   500    |   134.0649   |      -       |                   |
|   29   |    GAN    |     FSRCNN      |  Bilinear   |    19 out    |     -      |  -   |    -     |   127.3791   |      -       |                   |
|   30   |    GAN    |    U-Net 256    |  Bilinear   |    20 out    |     64     | 2e-4 |   400    |   128.4035   |      -       |     Synthetic     |
|   31   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   200    |   135.3882   |      -       |                   |
|   32   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   300    |   134.9906   |      -       |                   |
|   33   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   500    |   120.9798   |   122.2001   |                   |
|   34   |    GAN    |       19        |  Bilinear   |   Separate   |     -      |  -   |    -     |   219.0236   |      -       |  Denoise filter   |
|   35   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   550    |   124.6969   |      -       |                   |
|   36   |    GAN    |   U-Net 256/L   |  Bilinear   |   Separate   |     64     | 2e-4 |   400    |   124.8068   |      -       |                   |
|   37   |    GAN    |    U-Net SA     |  Bilinear   |   Separate   |    256     | 2e-4 |   600    |   132.8082   |      -       |                   |
|   38   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     36     | 2e-4 |   500    |   123.1196   |      -       |                   |
|   39   |    GAN    |  U-Net 256/XL   |  Bilinear   |   Separate   |     32     | 2e-4 |   400    |   116.0613   |   115.7694   |                   |
|   40   |    GAN    |    U-Net SA     |  Bilinear   |   Separate   |    288     | 2e-4 |   500    |   151.7567   |   153.0398   |                   |
|   41   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    144     | 2e-4 |   500    |   123.3888   |   125.5579   |  Inception loss   |
|   42   |    GAN    |       39        |   Lanczos   |   Separate   |     -      |  -   |    -     |   117.4499   |   117.1678   |   Detail filter   |
|   43   |    GAN    |  U-Net 256/XL   |  Bilinear   |   Separate   |     32     | 2e-4 |   450    |   131.0206   |   127.2280   |                   |
|   44   |    GAN    |       39        |   Lanczos   |   Separate   |     -      |  -   |    -     |   116.4546   |   115.5285   |                   |
|   45   |    GAN    |  U-Net 256/XL   |  Bilinear   |   Separate   |     32     | 2e-4 |   500    |   116.1229   |   117.0757   |                   |
|   46   |    GAN    |       39        |   Bicubic   |   Separate   |     -      |  -   |    -     |   116.0382   |   115.2611   |                   |
|   47   | Diffusion |      PITI       |   Bicubic   |   Separate   |     4      | 1e-5 | 2e4 step |   118.9399   |   117.7538   |                   |
| **48** | **Mixed** |   **47 + 39**   | **Bicubic** | **Separate** |     -      |  -   |    -     | **106.2478** | **104.2643** | **8MB threshold** |
|   49   |   Mixed   |     47 + 39     |   Bicubic   |   Separate   |     -      |  -   |    -     |   108.2726   |   105.6473   |   Maximum size    |
|   50   | Diffusion |       47        |   Bicubic   |   Separate   |     -      |  -   |    -     |   108.5719   |   112.4611   |    Sample_c 4     |
|   51   | Diffusion |      PITI       |   Bicubic   |   Separate   |     4      | 1e-5 | 2e4 step |   115.9617   |   117.3084   |    Fill image     |
|   52   | Diffusion |       47        |   Bicubic   |   Separate   |     -      |  -   |    -     |   109.5812   |   111.4152   |   Respacing 500   |
