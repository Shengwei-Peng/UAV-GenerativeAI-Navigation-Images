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
        os.makedirs(f"{args.output_folder}/RO/{mode}/label_img", exist_ok=True)
        os.makedirs(f"{args.output_folder}/RO/{mode}/img", exist_ok=True)
        os.makedirs(f"{args.output_folder}/RI/{mode}/label_img", exist_ok=True)
        os.makedirs(f"{args.output_folder}/RI/{mode}/img", exist_ok=True)
        RO_img_paths = []
        RI_img_paths = []
        for label_img_file, img_file in progress_bar:
            label_img_path = f"{data_folder}/label_img/{label_img_file}"
            img_path = f"{data_folder}/img/{img_file}" if train else label_img_path
            with Image.open(label_img_path) as label_img, Image.open(img_path) as img:
                img = img.resize((256, 256))
                label_img = label_img.resize((256, 256))
                category = "RO" if "RO" in img_file else "RI"
                label_img_save_path = f"{args.output_folder}/{category}/{mode}/label_img/{label_img_file.replace('png', 'jpg')}"
                img_save_path = f"{args.output_folder}/{category}/{mode}/img/{img_file.replace('png', 'jpg')}"
                label_img.save(label_img_save_path)
                img.save(img_save_path)
                absolute_path = os.path.abspath(img_save_path)
                if category == "RO":
                    RO_img_paths.append(absolute_path)
                else:
                    RI_img_paths.append(absolute_path)
                progress_bar.set_description(f"Processing {label_img_file} and {img_file}")
        with open(f"{args.output_folder}/RO_{mode}.txt", 'w') as file:
            for path in RO_img_paths:
                file.write(path + '\n')
        with open(f"{args.output_folder}/RI_{mode}.txt", 'w') as file:
            for path in RI_img_paths:
                file.write(path + '\n')      
        train = False

if __name__ == '__main__':
    args = parse_args()
    pre_process(args)