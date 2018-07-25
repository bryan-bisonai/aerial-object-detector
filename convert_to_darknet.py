import os
import numpy as np

import data_utils as util

'''
Convert DOTA format to YOLO(darknet) required
'''


def dota2darknet(src, dst, classnames):
    """
    :param src: path to txt in DOTA format
    :param dst: path to txt in YOLO format
    :param classnames: selected categories
    :return: creates files that contain converted data set in YOLO format
    """
    filelist = util.GetFileFromThisRootDir(src, 'txt')
    for idx, path in enumerate(filelist):
        objects = util.parse_dota_poly(path)
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(dst, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(util.dots4ToRecC(poly, img_w=1024, img_h=1024))
                if obj['name'] in classnames:
                    id = classnames.index(obj['name'])
                else:
                    print('[error] class is not in the list: [{}]'.format(obj['name']))
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
                f_out.write(outline + '\n')


if __name__ == '__main__':
    dota2darknet(util.d_train.label_dir, '/data/public/rw/datasets/DOTA/train/labelTxtYOLO', util.dota_15)
