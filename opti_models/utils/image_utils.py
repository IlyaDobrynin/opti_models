import os
from tqdm import tqdm
import cv2
from PIL import Image
from ido_cv import draw_images


def convert_images(path_to_images: str, to_postfix: str = '.png', out_path: str = ''):


    for item in tqdm(os.walk(path_to_images)):
        root_path = item[0]
        in_folders = item[1]
        in_files = item[2]
        if len(in_files) > 0:
            folder_name = root_path.split("/")[-1]
            out_folder = os.path.join(out_path, folder_name)
            os.makedirs(out_folder, exist_ok=True)
            for img_name in in_files:
                image_jpeg = os.path.join(root_path, img_name)
                image = Image.open(image_jpeg)
                image.save(os.path.join(out_folder, f'{img_name[:-5]}.png'))
                # draw_images([image])

    # images_folder_path = os.path.join(path_to_images, str(img_cls))
    # for img_name in os.listdir(images_folder_path):
    #     path_to_image = os.path.join(images_folder_path, img_name)
    #     out_dict[path_to_image] = img_cls
    #     if show:
    #         image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    path_to_images = "/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
    out_path = "/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2_topimages_png"
    convert_images(path_to_images=path_to_images, out_path=out_path)