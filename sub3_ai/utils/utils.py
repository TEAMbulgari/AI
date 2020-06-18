from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import tensorflow as tf

from config import config

# Req. 2-2	세팅 값 저장
def save_config():
	pass


# Req. 4-1	이미지와 캡션 시각화
def visualize_img_caption(img_path, caption):
    
    image = img.imread(config.images_file_path + img_path[0])
    
    plt.title(caption[0])
    plt.imshow(image)