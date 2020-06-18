

from pytest import run_inference_on_image



# 동물 테스트
imagePath = './img/023.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = './model/model_5000.pb'                             # 읽어들일 graph 파일 경로
labelsFullPath = './model/labels_5000.txt'                               # 읽어들일 labels 파일 경로
# modelFullPath = './model/saved_model.pb'                             # 읽어들일 graph 파일 경로
# labelsFullPath = './model/labels_0424.txt'                               # 읽어들일 labels 파일 경로

animal_result = run_inference_on_image(imagePath, modelFullPath, labelsFullPath)
print(animal_result)