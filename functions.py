import numpy as np
import pandas as pd
import os
import PIL
from PIL import Image
from numpy import asarray
import pathlib as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from pycocotools.coco import COCO
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import (
    img_to_array,
    array_to_img,
    ImageDataGenerator,
)
import skimage.io as io
import random
import cv2
from itertools import product
from fastai.vision.all import *


def sub_plots(number, x_small, plot_title):
    label_dict = {"1": "Label", "2": "Before", "3": "After"}
    np_before = asarray(Image.open(f"10x_before/site{number}.jpg"))
    np_after = asarray(Image.open(f"10x_after/site{number}.jpg"))
    np_label = np.subtract(np_after, np_before)
    a = Image.open(f"10x_before/site{number}.jpg")
    height, width = a.shape
    plt.suptitle(plot_title)
    plt.subplots(figsize=(10, 5))
    for i, j in zip((np_label, np_before, np_after), (1, 2, 3)):
        ax = plt.subplot(1, 3, j)
        plt.imshow(cv2.resize(i, (width // x_small, height // x_small)))
        plt.title(label_dict[str(j)])
    plt.show()


def empty_mask(image_dir, mask_dir):
    fnames = get_image_files(image_dir)
    mask = np.zeros((1944, 2594))
    for i in range(len(fnames)):
        file_name = str(fnames[i]).split("/")[-1]
        file_name = file_name.split(".")[0]
        out = os.path.join(mask_dir, "mask_{}.png".format(file_name))
        cv2.imwrite(out, (255 * mask).astype(np.uint8))


def create_mask(mask_dir, coco, catIDs):

    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    annIds = coco.getAnnIds(imgIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for i in range(len(anns)):
        name = images[i]["file_name"]
        name = name.split(".")[0]
        out = os.path.join(mask_dir, "mask_{}.png".format(name))
        mask = coco.annToMask(anns[i])
        cv2.imwrite(out, mask.astype(np.uint16))


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    img = img.crop((318, 0, 2262, 1944))
    w, h = img.size

    grid = list(product(range(0, h - h % d, d), range(0, w - w % d, d)))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f"{name}_{i}_{j}{ext}")
        img.crop(box).save(out)


def image_names(image_dir):
    fnames = get_image_files(image_dir)
    names = []
    for i in range(len(fnames)):
        names.append(str(fnames[i]).split("/")[-1])

    return names


def get_image_filter(path, not_valid):
    fnames = get_image_files(path)
    fnames2 = get_image_files(path)
    b = not_valid["Site_nr"].copy()
    for i in fnames:
        if i.stem.split("_")[0] in list(b):
            fnames2.remove(i)
    return fnames2


def get_mask(path, y, folder="crop_mask"):
    return os.path.join(path, folder, f"mask_{y.stem}.png")


def get_mask2(path, y, input_image_size=(324, 324), folder="crop_mask"):
    img = os.path.join(path, folder, f"mask_{y.stem}.png")
    mask0 = asarray(Image.open(img))
    mask1 = np.zeros(input_image_size)
    pixel_value = 2
    new_mask = cv2.resize(mask0 * pixel_value, input_image_size)
    mask = np.maximum(new_mask, mask1)
    # mask = mask.reshape(input_image_size[0], input_image_size[1], 1)
    return mask

