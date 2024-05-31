echo "[GAN] Preprocessing data..."
python ./gan/preprocess.py \
  --train_data_folder="./training_dataset" \
  --test_data_folder="./testing_dataset" \
  --output_folder="./gan/datasets"


echo "[GAN] Training for river data..."
python ./gan/train.py \
  --seed=0 \
  --model=pix2pix \
  --direction=AtoB \
  --netG=unet_256 \
  --ngf=256 \
  --ndf=256 \
  --dataroot="./gan/datasets/RI" \
  --name="gan/RI" \
  --batch_size=32 \
  --n_epochs=1 \
  --n_epochs_decay=1


echo "[GAN] Training for road data..."
python ./gan/train.py \
  --seed=0 \
  --model=pix2pix \
  --direction=AtoB \
  --netG=unet_256 \
  --ngf=256 \
  --ndf=256 \
  --dataroot="./gan/datasets/RO" \
  --name="gan/RO" \
  --batch_size=32 \
  --n_epochs=1 \
  --n_epochs_decay=1


echo "[GAN] Testining for river data..."
python ./gan/test.py \
  --seed=0 \
  --model=pix2pix \
  --direction=AtoB \
  --dataroot="./gan/datasets/RI" \
  --netG=unet_256 \
  --ngf=256 \
  --name=gan/RI


echo "[GAN] Testining for river data..."
python ./GAN/test.py \
  --seed=0 \
  --model=pix2pix \
  --direction=AtoB \
  --dataroot="./GAN/datasets/RO" \
  --netG=unet_256 \
  --ngf=256 \
  --name=gan/RO
