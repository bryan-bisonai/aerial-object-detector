import sys
import os
import re
import codecs
import base64
import imghdr

import math
import numpy as np
import shapely.geometry as shgeo

from PIL import Image

dota_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
           'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
           'swimming-pool', 'helicopter']


class d_train:
    img_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/train/images/images'
    img_dir_jpg = '/data/public/rw/datasets/aerial_inspection/DOTA/train/images/images_jpg'
    label_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/train/labelTxt'


class d_valid:
    img_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/valid/images'
    img_dir_jpg = '/data/public/rw/datasets/aerial_inspection/DOTA/valid/images_jpg'
    label_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/valid/labelTxt'


def get_file_from_this_rootdir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext is not None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
            else:  # TODO: possible bugs?
                pass
            
    return allfiles


def parse_dota_poly(filename):
    """
    Parse DOTA gound truth in the format:
    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    f = []
    if sys.version_info >= (3, 5):
        fd = open(filename, 'r')
        f = fd
    elif sys.version_info >= 2.7:
        fd = codecs.open(filename, 'r')
        f = fd
    while True:  # TODO: possible bugs?
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}

            # Create a reformatted dict for DOTA format
            if len(splitlines) < 9:
                continue
            if len(splitlines) >= 9:
                object_struct['name'] = splitlines[8]
            if len(splitlines) == 9:
                object_struct['difficult'] = '0'
            elif len(splitlines) >= 10:
                object_struct['difficult'] = splitlines[9]

            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            objects.append(object_struct)
        else:
            break

    return objects


def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))),\
                             max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))),\
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))),\
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))

    return xmin, xmax, ymin, ymax


def dots4ToRecC(poly, img_w, img_h):
    xmin, xmax, ymin, ymax = dots4ToRec4(poly)
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return x/img_w, y/img_h, w/img_w, h/img_h


def encode_image_png(filepath):
    with open(filepath, 'rb') as imageFile:
        b_img = base64.b64encode(imageFile.read())
    return b_img


def convert_to_jpeg(file, save_dir):
    file_type = imghdr.what(file)
    if file_type != 'jpeg':
        img = Image.open(file)
        img.convert('RGB').save(save_dir + os.path.splitext(os.path.basename(file))[0] + '.jpg', 'JPEG')


if __name__ == '__main__':
    pass
