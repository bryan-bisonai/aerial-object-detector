from tqdm import tqdm

from data_utils import convert_to_jpeg, get_file_from_this_rootdir


# source_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/train/images/images'
source_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/valid/images'
# save_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/train/images/images_jpg/'
save_dir = '/data/public/rw/datasets/aerial_inspection/DOTA/valid/images_jpg/'


if __name__ == '__main__':
    filepath_list = get_file_from_this_rootdir(source_dir, 'png')
    for file in tqdm(filepath_list):
        convert_to_jpeg(file, save_dir)
    print('covert image type to JPEG - Done!!!!!')
