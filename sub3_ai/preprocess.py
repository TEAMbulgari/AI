from config import config
import cv2
import glob
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import random
import face_recognition as fr
import os

# 이미지 전처리(Actor)
def setPreprocessing():
    # Req. 1-1    이미지 파일 로드
    # globImage = glob.glob(config.images_file_path_actor+"*.jpg")
    globActorImgFolders = glob.glob(config.images_file_path_actor+"*")
    globSingerImgFolders = glob.glob(config.images_file_path_singer+"*")
   
    for i in range(len(globActorImgFolders)):
    
        paths = globActorImgFolders[i][54:]
        
        # 각 폴더마다 이미지 파일 가져오기
        globActorImage = glob.glob(globActorImgFolders[i]+"\*.jpg")
        known_person_list = []
        for k in range(len(globActorImage)):
            globActorImage[k] = globActorImage[k].replace("\\","/")
            known_person_list.append(fr.load_image_file(globActorImage[k]))
        
        
        # 사진마다 얼굴 인식하기
        known_face_list = []
        for person in known_person_list:
            locations = fr.face_locations(person)
            if(locations):
                top, right, bottom, left = locations[0]
                face_image = person[top:bottom, left:right]
                known_face_list.append(face_image)
        
    
        # 이미지 리사이징 작업
        resizeImage = []        # 리사이징 된 이미지리스트 선언
        normalizeImage = []     # 정규화한 이미지
        for j in range (len(known_face_list)):
            # img = cv2.imread(known_face_list[j])
            img = known_face_list[j]
            height, width, rgb = img.shape  # 이미지 크기 확인
            
            # openCV는 BGR로 사용하지만, Matplotlib는 RGB로 이미지를 보여주기 때문
            b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
            img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
            
            resizeImage.append(cv2.resize(img2, dsize=(300,300))) # 이미지 리사이징
             
            # Req. 1-2    이미지 정규화
            normalizeImage.append(cv2.normalize(resizeImage[j], None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))    
            
            # 파일저장
            directory = './datasets/resembleImages/'+paths
            file = './datasets/resembleImages/'+paths+'/'+str(j)+'.jpg'
            file = file.replace("\\","/")
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            cv2.imwrite(file, normalizeImage[j])
            
            
            
# 이미지 전처리(Singer)
def setPreprocessing2():
    # Req. 1-1    이미지 파일 로드
    # globImage = glob.glob(config.images_file_path_actor+"*.jpg")
    globSingerImgFolders = glob.glob(config.images_file_path_singer+"*")
   
    for i in range(len(globSingerImgFolders)):
    
        paths = globSingerImgFolders[i][54:]
        
        # 각 폴더마다 이미지 파일 가져오기
        globActorImage = glob.glob(globSingerImgFolders[i]+"\*.jpg")
        known_person_list = []
        for k in range(len(globActorImage)):
            globActorImage[k] = globActorImage[k].replace("\\","/")
            known_person_list.append(fr.load_image_file(globActorImage[k]))
        
        
        # 사진마다 얼굴 인식하기
        known_face_list = []
        for person in known_person_list:
            locations = fr.face_locations(person)
            if(locations):
                top, right, bottom, left = locations[0]
                face_image = person[top:bottom, left:right]
                known_face_list.append(face_image)
        
    
        # 이미지 리사이징 작업
        resizeImage = []        # 리사이징 된 이미지리스트 선언
        normalizeImage = []     # 정규화한 이미지
        for j in range (len(known_face_list)):
            # img = cv2.imread(known_face_list[j])
            img = known_face_list[j]
            height, width, rgb = img.shape  # 이미지 크기 확인
            
            # openCV는 BGR로 사용하지만, Matplotlib는 RGB로 이미지를 보여주기 때문
            b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
            img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
            
            resizeImage.append(cv2.resize(img2, dsize=(300,300))) # 이미지 리사이징
             
            # Req. 1-2    이미지 정규화
            normalizeImage.append(cv2.normalize(resizeImage[j], None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))    
            
            # 파일저장
            directory = './datasets/resembleImages/'+paths
            file = './datasets/resembleImages/'+paths+'/'+str(j)+'.jpg'
            file = file.replace("\\","/")
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            cv2.imwrite(file, normalizeImage[j])