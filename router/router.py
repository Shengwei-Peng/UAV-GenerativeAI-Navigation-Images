import os
import argparse
from PIL import Image
from tqdm import tqdm 
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_results_folder", type=str, required=True)
    parser.add_argument("--gan_results_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()
    return args

def post_process(args):
    diff_folder = args.diffusion_results_folder
    gan_folder = args.gan_results_folder
    os.makedirs(args.output_folder, exist_ok=True)
    diff_images = {file_name for file_name in os.listdir(diff_folder) if os.path.isfile(os.path.join(diff_folder, file_name))}
    gan_images = {file_name for file_name in os.listdir(gan_folder) if os.path.isfile(os.path.join(gan_folder, file_name))}
    common_image_names = diff_images.intersection(gan_images)
    for image_name in tqdm(common_image_names, desc="Processing images"):
        diff_image_path = os.path.join(diff_folder, image_name)
        gan_image_path = os.path.join(gan_folder, image_name)
        with Image.open(diff_image_path) as diff_img, Image.open(gan_image_path) as gan_img:
            gan_img = gan_img.resize((428, 240), Image.BICUBIC)
            diff_img = diff_img.resize((428, 240), Image.BICUBIC)
        gan_img.save(gan_image_path)
        diff_img.save(diff_image_path)
        size_diffusion = os.path.getsize(diff_image_path)
        if size_diffusion > 8 * 1024:
            diff_img.save(os.path.join(args.output_folder, image_name))
        else:
            gan_img.save(os.path.join(args.output_folder, image_name))
    shutil.make_archive(args.output_folder, 'zip', args.output_folder)

if __name__ == '__main__':
    args = parse_args()
    post_process(args)
