# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import face_recognition as fr
import cv2
import argparse
import os
import numpy as np

from PIL import Image

import tensorflow as tf # TF2


def load_labels(filename):
  with open(filename, 'r', encoding='cp949', errors='ignore') as f:
    return [line.strip() for line in f.readlines()]


def execute_model(filepath, label_file,  input_mean, input_std):
#if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument(
  #     '-i',
  #     '--image',
  #     default='/tmp/grace_hopper.bmp',
  #     help='image to be classified')
  # parser.add_argument(
  #     '-m',
  #     '--model_file',
  #     default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
  #     help='.tflite model to be executed')
  # parser.add_argument(
  #     '-l',
  #     '--label_file',
  #     default='/tmp/labels.txt',
  #     help='name of file containing labels')
  # parser.add_argument(
  #     '--input_mean',
  #     default=127.5, type=float,
  #     help='input_mean')
  # parser.add_argument(
  #     '--input_std',
  #     default=127.5, type=float,
  #     help='input standard deviation')
  # args = parser.parse_args()

  # interpreter = tf.lite.Interpreter(model_path=args.model_file)
  interpreter = tf.lite.Interpreter(model_path="sub3_ai/new_mobile_model.tflite")
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  
  
  # 사진마다 얼굴 인식하기
  #face_image=fr.load_image_file('./'+args.image)
  face_image=fr.load_image_file(filepath)
  locations = fr.face_locations(face_image)
  if(locations):
      top, right, bottom, left = locations[0]
      face_image = face_image[top:bottom, left:right]
  
  #img = face_image.resize((width, height))
  #b, g, r = cv2.split(face_image)   # img파일을 b,g,r로 분리
  #face_image = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
  img = cv2.resize(face_image, dsize=(width,height)) # 이미지 리사이징
             
  img = cv2.normalize(img, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)   
            

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    #input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  #labels = load_labels(args.label_file)
  labels = load_labels(label_file)
  
  res1 = []
  res2 = []
  
  for i in top_k:
    if floating_model:
      #print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      res1.append(float(results[i]))
      labels[i] = labels[i][0:labels[i].find("(")]  
      res2.append(labels[i])
    else:
      #print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
      res1.append(float(results[i] / 255.0))
      labels[i] = labels[i][0:labels[i].find("(")]
      res2.append(labels[i])
      

  return res1, res2