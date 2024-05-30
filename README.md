# Generative AI Navigation Information for UAV Reconnaissance in Natural Environments
## Introduction
This project leverages generative AI to create realistic environmental data for UAV reconnaissance, enhancing the efficiency and accuracy of terrain and natural environment surveys. The focus is on generating high-quality images of roads and rivers from UAV perspectives under various conditions.
## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments.git
cd Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments
```
Install the required packages:
```bash
pip install -r requirements.txt
```
## Usage
To start generating images, run the following command:
### Preprocessing
Prepare the data for training and testing:
```bash
python preprocess.py \
  --train_data_folder="./34_Competition_1_Training_dataset/Training_dataset" \
  --test_data_folder="./34_Competition_1_public_testing_dataset" \
  --output_folder="./aicup_dataset"
```
- `train_data_folder`: Required. The path to the folder containing training data files.
- `test_data_folder`: Required. The path to the folder containing testing data files.
- `output_folder`: Required. The path to the folder for saving processed data.

### Training
Train the model using the preprocessed data:
```bash
python train.py  \
  --dataroot="./aicup_dataset"  \
  --name=aicup_pix2pix  \
  --model=pix2pix  \
  --direction=AtoB
```
### Testing
Evaluate the model using the test dataset:
```bash
python test.py \
  --dataroot="./aicup_dataset"  \
  --name=aicup_pix2pix  \
  --model=pix2pix  \
  --direction=AtoB
```

## Acknowledgement
We extend our gratitude to the developers of [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [PITI](https://github.com/PITI-Synthesis/PITI) for generously sharing their code, which has been invaluable to our work.

## Results
Below is a table showcasing the results of image generation for different environments:
|   ID   |  Method   |    Generator    |   Resize    |  Data Type   | Batch Size |  LR  |  Epochs  |   PUB FID    |   PRI FID    |      Note      |
|:------:|:---------:|:---------------:|:-----------:|:------------:|:----------:|:----:|:--------:|:------------:|:------------:|:--------------:|
|   01   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   200    |   149.5899   |      -       |                |
|   02   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    256     | 2e-4 |   200    |   163.8351   |      -       |                |
|   03   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   400    |   133.0080   |      -       |                |
|   04   |    GAN    | ResNet 9blocks  |  Bilinear   |    Mixed     |     64     | 2e-4 |   200    |   267.8923   |      -       |                |
|   05   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   400    |   133.7452   |      -       |     Extra      |
|   06   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   600    |   129.6689   |      -       |                |
|   07   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |    256     | 2e-4 |   1000   |   134.3076   |      -       |                |
|   08   |    GAN    |      U-Net      |  Bilinear   |    Mixed     |     16     | 2e-4 |   200    |   137.9879   |      -       |                |
|   09   |    GAN    |    U-Net 512    |  Bilinear   |    Mixed     |     16     | 2e-4 |   200    |   141.4001   |      -       |                |
|   10   |    GAN    |      U-Net      |  Bilinear   |    Mixed     |     64     | 2e-4 |   600    |   142.0793   |      -       |                |
|   11   |    GAN    |    U-Net 256    |  Bilinear   |    Mixed     |     64     | 2e-4 |   200    |   135.5488   |      -       |                |
|   12   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   400    |   127.8701   |      -       |                |
|   13   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   600    |   156.7936   |      -       |                |
|   14   |    GAN    |       VAE       |  Bilinear   |    Mixed     |     8      | 2e-4 |   200    |   133.0856   |      -       |                |
|   15   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     1      | 2e-4 |   200    |   206.0832   |      -       |                |
|   16   | Diffusion | SDv1-5 Scribble |  Bilinear   |    Mixed     |     8      | 1e-5 | 1e4 step |   186.8134   |      -       |                |
|   17   | Diffusion | SDv1-5 Softedge |  Bilinear   |    Mixed     |     8      | 1e-5 | 1e4 step |   211.2076   |      -       |                |
|   18   |    GAN    |    U-Net 256    |  Bilinear   |    12 out    |     64     | 2e-4 |   400    |   127.1882   |      -       |   Synthetic    |
|   19   |    GAN    |    U-Net 256    |  Bilinear   |    18 out    |     64     | 2e-4 |   400    |   123.7632   |      -       |   Synthetic    |
|   20   |    GAN    |    U-Net 256    |  Bilinear   |    19 out    |     64     | 2e-4 |   400    |   124.1638   |      -       |   Synthetic    |
|   21   |    GAN    |    U-Net 512    |  Bilinear   |   Separate   |     16     | 2e-4 |   400    |   136.8711   |      -       |                |
|   22   | Diffusion | SDv1-5 Scribble |  Bilinear   |    Mixed     |     64     | 1e-5 | 1e4 step |   190.1620   |      -       |                |
|   23   |   Mixed   |     19 + 22     |  Bilinear   |   Separate   |     -      |  -   |    -     |   144.5344   |      -       |                |
|   24   | Diffusion |       22        |  Bilinear   |    Mixed     |     -      |  -   |    -     |   192.3814   |      -       | FP32 inference |
|   25   | Diffusion | SDv1-5 Scribble |  Bilinear   |    Mixed     |     64     | 1e-5 | 2e4 step |   200.9103   |      -       |                |
|   26   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    256     | 2e-4 |   400    |   130.6916   |      -       |                |
|   27   |    GAN    |     FSRCNN      |  Bilinear   |    19 out    |     -      |  -   |    -     |   153.7523   |      -       | Denoise filter |
|   28   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    256     | 2e-4 |   500    |   134.0649   |      -       |                |
|   29   |    GAN    |     FSRCNN      |  Bilinear   |    19 out    |     -      |  -   |    -     |   127.3791   |      -       |                |
|   30   |    GAN    |    U-Net 256    |  Bilinear   |    20 out    |     64     | 2e-4 |   400    |   128.4035   |      -       |   Synthetic    |
|   31   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   200    |   135.3882   |      -       |                |
|   32   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   300    |   134.9906   |      -       |                |
|   33   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   500    |   120.9798   |   122.2001   |                |
|   34   |    GAN    |       19        |  Bilinear   |   Separate   |     -      |  -   |    -     |   219.0236   |      -       | Denoise filter |
|   35   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     64     | 2e-4 |   550    |   124.6969   |      -       |                |
|   36   |    GAN    |   U-Net 256/L   |  Bilinear   |   Separate   |     64     | 2e-4 |   400    |   124.8068   |      -       |                |
|   37   |    GAN    |    U-Net SA     |  Bilinear   |   Separate   |    256     | 2e-4 |   600    |   132.8082   |      -       |                |
|   38   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |     36     | 2e-4 |   500    |   123.1196   |      -       |                |
|   39   |    GAN    |  U-Net 256/XL   |  Bilinear   |   Separate   |     32     | 2e-4 |   400    |   116.0613   |   115.7694   |                |
|   40   |    GAN    |    U-Net SA     |  Bilinear   |   Separate   |    288     | 2e-4 |   500    |   151.7567   |   153.0398   |                |
|   41   |    GAN    |    U-Net 256    |  Bilinear   |   Separate   |    144     | 2e-4 |   500    |   123.3888   |   125.5579   | Inception loss |
|   42   |    GAN    |       39        |   Lanczos   |   Separate   |     -      |  -   |    -     |   117.4499   |   117.1678   | Detail filter  |
|   43   |    GAN    |  U-Net 256/XL   |  Bilinear   |   Separate   |     32     | 2e-4 |   450    |   131.0206   |   127.2280   |                |
|   44   |    GAN    |       39        |   Lanczos   |   Separate   |     -      |  -   |    -     |   116.4546   |   115.5285   |                |
|   45   |    GAN    |  U-Net 256/XL   |  Bilinear   |   Separate   |     32     | 2e-4 |   500    |   116.1229   |   117.0757   |                |
|   46   |    GAN    |       39        |   Bicubic   |   Separate   |     -      |  -   |    -     |   116.0382   |   115.2611   |                |
|   47   | Diffusion |      PITI       |   Bicubic   |   Separate   |     4      | 1e-5 | 2e4 step |   118.9399   |   117.7538   |                |
| **48** | **Mixed** |   **47 + 39**   | **Bicubic** | **Separate** |     -      |  -   |    -     | **106.2478** | **104.2643** | 8MB threshold  |
|   49   |   Mixed   |     47 + 39     |   Bicubic   |   Separate   |     -      |  -   |    -     |   108.2726   |   105.6473   |  Maximum size  |
|   50   | Diffusion |       47        |   Bicubic   |   Separate   |     -      |  -   |    -     |   108.5719   |   112.4611   |   Sample_c 4   |
|   51   | Diffusion |      PITI       |   Bicubic   |   Separate   |     4      | 1e-5 | 2e4 step |   115.9617   |   117.3084   |   Fill image   |
|   52   | Diffusion |       47        |   Bicubic   |   Separate   |     -      |  -   |    -     |   109.5812   |   111.4152   | Respacing 500  |
