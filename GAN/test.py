import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from models import create_model
from data import create_dataset
from options.test_options import TestOptions
from util.util import set_seed


def save_image(visuals, output_path):
    image = visuals["fake_B"][0]
    image = image.cpu().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(output_path)


def main():
    opt = TestOptions().parse()
    set_seed(seed=opt.seed)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    output_folder = f"./results/{opt.name[:-3]}"
    os.makedirs(output_folder, exist_ok=True)

    model.eval()
    for data in tqdm(dataset, desc="Processing images"):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()[0].split('/')[-1].replace(".png", ".jpg")
        output_path = f"{output_folder}/{img_path}"
        save_image(visuals, output_path)


if __name__ == '__main__':
    main()
