from config import config
from data import preprocess
import numpy as np
import pandas as pd

# 총 이미지 수  = 31783
# 총 캡션 수 = 158915
# 토크나이저 갯수 = 18316

# 정규화 된 이미지 딕셔너리 저장 (3만장)
img_dict = preprocess.get_load_image()

# 토크나이저 저장
train_seqs = preprocess.create_tokenizer(10000)

# 토크나이저 불러오기
caption_token = preprocess.load_tokenizer()

# 쌍으로 묶기 (주석풀기)
# couple = pd.read_csv(config.caption_file_path, encoding='cp949', sep='|', quoting=3)

# ######################################################################
# 임시테스트 10개
temp_path = "C:/Users/multicampus/Desktop/ai304/datasets/caption10.csv"
couple = pd.read_csv(temp_path, encoding='cp949', sep='|', quoting=3)
train_seqs = caption_token.texts_to_sequences(couple[' comment,,,'])
# ######################################################################

couple["tkcaption"] = train_seqs
#couple[index][2] = ''.join(str(train_seqs[index]))

# 이미지 전처리
couple['image_name'] = couple['image_name'].map(lambda x : x.replace('"',''))
couple["imgvector"] = couple['image_name'].map(lambda x: img_dict[x])


couple["imgvector"] = preprocess.augmentation(couple["imgvector"].tolist())
# augdata = preprocess.augmentation(couple["imgvector"].tolist())
