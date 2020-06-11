import argparse

# Config.py

parser = argparse.ArgumentParser()

parser.add_argument('--caption_file_path', type=str, default='./datasets/captions.csv')
parser.add_argument('--images_file_path', type=str, default='./datasets/images/')
parser.add_argument('--caption_file_path', type=str, default='./datasets/captions.csv')
parser.add_argument('--images_file_path_actor', type=str,
                    default='C:/Users/multicampus/Downloads/Extreme Picture Finder/Actor/')
parser.add_argument('--images_file_path_singer', type=str,
                    default='C:/Users/multicampus/Downloads/Extreme Picture Finder/Singer/')

config = parser.parse_args()

