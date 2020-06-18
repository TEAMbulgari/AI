from config import config
import cv2
import glob
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import random

# Req. 1 이미지 전처리
def get_load_image():
    # Req. 1-1	이미지 파일 로드
    globImage = glob.glob(config.images_file_path+"*.jpg")
    length = len(globImage) # 이미지 총 갯수
    length = 10 # 테스트용

    # 한개 이미지 테스트용
   # img = cv2.imread(globImage[0])
   # height, width, rgb = img.shape  # 이미지 크기 확인
   # resizeImage = cv2.resize(img, dsize=(300,300)) # 이미지 리사이징
    
   # cv2.imshow("img", resizeImage)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    
    
    # 이미지 리사이징 작업
    resizeImage = []        # 리사이징 된 이미지리스트 선언
    normalizeImage = []     # 정규화한 이미지
    for i in range (length):
        img = cv2.imread(globImage[i])
        height, width, rgb = img.shape  # 이미지 크기 확인
        resizeImage.append(cv2.resize(img, dsize=(300,300))) # 이미지 리사이징
         
        # Req. 1-2	이미지 정규화
        normalizeImage.append(cv2.normalize(resizeImage[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    
    dict_img = {}

    for i in range(length):        
        dict_img[globImage[i][18:]] = normalizeImage[i]

    return dict_img


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# Req. 2 텍스트 데이터 전처리
def create_tokenizer(top_k):
    
    # Req. 2-1 텍스트 데이터 토큰화
    datas = np.loadtxt(config.caption_file_path, delimiter ="|", dtype=str, skiprows=1)
    
    captions = []
    for annot in datas[:,2]:
        caption = '<start> ' + annot + ' <end>'
        captions.append(caption)
 
    
    # Choose the top 5000 words from the vocabulary
    # top_k = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    
    
    tokenizer.fit_on_texts(captions)    
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(captions)
    

    # max_length = calc_max_length(train_seqs) # 80 너무크네
    max_length = 30
    
    # 패딩
    train_seqs = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=max_length, padding='post', value=0).tolist()
    
    
    # Req. 2-2 tokenizer 저장 및 불러오기
    # save
    with open('./datasets/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)
    
    return train_seqs
    
# tokenizer 불러오기
def load_tokenizer():
    # load
    with open('./datasets/tokenizer.pickle', 'rb') as f:
        token = pickle.load(f)
    return token
    

# Req 3. Dataset 만들기
def create_dataset(imgList, captionToken):
    
    
    tfdata = tf.data.Dataset.from_tensor_slices(captionToken)
    
    return tfdata



def augmentation(tensor_image):
    
    ## 전체 이미지셋에서 랜덤 이미지를 추출하여 어그멘테이션 적용 후 다시 저장 후 리턴하
    # 랜덤 구현하기
    lengths = int(len(tensor_image) * 0.3)

    
    # random_images = random.sample(tensor_image, lengths)
    lists = random.sample(range(len(tensor_image)), lengths)
    
    for index in lists:
        tensor_image[index] = tf.keras.preprocessing.image.random_rotation(
            tensor_image[index], 180, row_axis=1, col_axis=2, channel_axis=2, fill_mode='nearest', cval=0.0,
            interpolation_order=1
        )
    
    
    '''
    ## TEST 용 #######################################################
    plt.figure(figsize=(10,5))
    i=0
    for index in lists:
        tensor_image[index] = tf.keras.preprocessing.image.random_rotation(
            tensor_image[index], 180, row_axis=1, col_axis=2, channel_axis=2, fill_mode='nearest', cval=0.0,
            interpolation_order=1
        )
        i += 1
        plt.subplot(2,5,i)
        plt.grid(False)
        plt.imshow(tensor_image[index], cmap=plt.cm.binary)
    plt.show()
    ## TEST 용 #######################################################
    '''
    
    return tensor_image




# os.getcwd() # 파일 저장되는 경로 확인 >> path

# csv_data = np.loadtxt(path, delimiter ="|", dtype=str, skiprows=1)


# train_data, validation_data= train_test_split(csv_data, test_size=0.30, random_state=321)

# np.savetxt(train_dataset_path, train_data ,delimiter=',', fmt = '%s')

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=sample_size, random_state=123)
