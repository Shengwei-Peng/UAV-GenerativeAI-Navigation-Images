import os
import argparse
from PIL import Image
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_folder", type=str, required=True)
    parser.add_argument("--test_data_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()
    return args

def pre_process(args):
    train = True
    for data_folder in [args.train_data_folder, args.test_data_folder]:
        label_imgs = sorted([f for f in os.listdir(f"{data_folder}/label_img") if f.endswith('.png')])
        imgs = sorted([f for f in os.listdir(f"{data_folder}/img") if f.endswith('.jpg')]) if train else label_imgs
        progress_bar = tqdm(zip(label_imgs, imgs), total=min(len(label_imgs), len(imgs)))
        mode = "train" if train else "test"
        os.makedirs(f"{args.output_folder}/RO/{mode}", exist_ok=True)
        os.makedirs(f"{args.output_folder}/RI/{mode}", exist_ok=True)
        for label_img_file, img_file in progress_bar:
            label_img_path = f"{data_folder}/label_img/{label_img_file}"
            img_path = f"{data_folder}/img/{img_file}" if train else label_img_path
            with Image.open(label_img_path) as label_img, Image.open(img_path) as img:
                total_width = label_img.width + img.width
                max_height = max(label_img.height, img.height)
                new_img = Image.new('RGB', (total_width, max_height))
                new_img.paste(label_img, (0, 0))
                new_img.paste(img, (img.width, 0))
                category = "RO" if "RO" in img_file else "RI"
                new_img.save(f"{args.output_folder}/{category}/{mode}/{img_file}")
                progress_bar.set_description(f"Processing {label_img_file} and {img_file}")
        train = False

if __name__ == '__main__':
    args = parse_args()
    pre_process(args)
