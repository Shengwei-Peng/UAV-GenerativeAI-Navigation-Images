echo "Downloading the best checkpoints"
gdown "https://drive.google.com/drive/folders/17B-PC-7MSnwbiohr4eGrcqg-U62wnHcE?usp=sharing" -O ./ --folder

echo "[GAN] Preprocessing data..."
python ./gan/preprocess.py \
  --train_data_folder="./training_dataset" \
  --test_data_folder="./testing_dataset" \
  --output_folder="./gan/datasets"


echo "[GAN] Testining for river data..."
python ./gan/test.py \
  --seed=0 \
  --model=pix2pix \
  --direction=AtoB \
  --dataroot="./gan/datasets/RI" \
  --netG=unet_256 \
  --ngf=256 \
  --name=best_gan/RI


echo "[GAN] Testining for river data..."
python ./gan/test.py \
  --seed=0 \
  --model=pix2pix \
  --direction=AtoB \
  --dataroot="./gan/datasets/RO" \
  --netG=unet_256 \
  --ngf=256 \
  --name=best_gan/RO


echo "[Diffusion] Preprocessing data..."
python ./diffusion/preprocess.py \
  --train_data_folder="./training_dataset" \
  --test_data_folder="./testing_dataset" \
  --output_folder="./diffusion/datasets"


echo "[Diffusion] Testining for river data..."
export LOGDIR=./results/best_diffusion/
MODEL_FLAGS="--learn_sigma True --model_path ./checkpoints/best_diffusion/RI/base/stage2-decoder/checkpoints/model020000.pt --sr_model_path ./checkpoints/best_diffusion/RI/upsample/checkpoints/model020000.pt"
SAMPLE_FLAGS="--num_samples 720 --sample_c 1.3 --batch_size 2"
DATASET_FLAGS="--data_dir ./diffusion/datasets/RI_train.txt --val_data_dir ./diffusion/datasets/RI_test.txt --mode coco-edge"
python ./diffusion/test.py $MODEL_FLAGS $SAMPLE_FLAGS  $DATASET_FLAGS


echo "[Diffusion] Testining for road data..."
export LOGDIR=./results/best_diffusion/
MODEL_FLAGS="--learn_sigma True --model_path ./checkpoints/best_diffusion/RO/base/stage2-decoder/checkpoints/model020000.pt --sr_model_path ./checkpoints/best_diffusion/RO/upsample/checkpoints/model020000.pt"
SAMPLE_FLAGS="--num_samples 720 --sample_c 1.3 --batch_size 2"
DATASET_FLAGS="--data_dir ./diffusion/datasets/RO_train.txt --val_data_dir ./diffusion/datasets/RO_test.txt --mode coco-edge"
python ./diffusion/test.py $MODEL_FLAGS $SAMPLE_FLAGS  $DATASET_FLAGS


echo "[Router] Selecting from the results..."
python ./router/router.py \
  --diffusion_results_folder="./results/best_diffusion" \
  --gan_results_folder="./results/best_gan" \
  --output_folder="./best_submission"
