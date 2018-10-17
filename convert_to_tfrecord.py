# ref: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
import os
import tensorflow as tf
import data_utils as util

from PIL import Image
from tqdm import tqdm
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

TYPE_OF_DATASET = 'valid'


if TYPE_OF_DATASET == 'train':
    source_img_dir = util.d_train.img_dir
    source_label_dir = util.d_train.label_dir
elif TYPE_OF_DATASET == 'valid':
    source_img_dir = util.d_valid.img_dir
    source_label_dir = util.d_valid.label_dir
else:
    print('[warning] Type of dataset is not defined')


def create_tf_example(name):
    # TODO(user): Populate the following variables from your example.
    b_image = util.encode_image_png(os.path.join(source_img_dir, name) + '.png')
    label_objects = util.parse_dota_poly(os.path.join(source_label_dir, name) + '.txt')

    width, height = Image.open(os.path.join(source_img_dir, name) + '.png').size  # Image width, height
    filename = name.encode()  # Filename of the image. Empty if image is not from file
    encoded_image_data = b_image  # Encoded image bytes
    image_format = b'png'  # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for obj in label_objects:
        poly = obj['poly']
        xmin, xmax, ymin, ymax = util.dots4ToRec4(poly)
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(obj['name'].encode())
        classes.append(util.dota_15.index(obj['name']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter('./datasets/{}_dataset.record'.format(TYPE_OF_DATASET))

    img_paths = util.get_file_from_this_rootdir(source_img_dir, 'jpg')
    for idx, path in enumerate(tqdm(img_paths)):
        name = os.path.splitext(os.path.basename(path))[0]
        tf_example = create_tf_example(name)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
