

from human_pytest import run_inference_on_image_human



## 사람 테스트
imagePath_human = './img/023.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath_human = './model/human_model6_30000.pb'                             # 읽어들일 graph 파일 경로
labelsFullPath_human = './model/labels_human6_30000.txt'                               # 읽어들일 labels 파일 경로

resultimage = run_inference_on_image_human(imagePath_human, modelFullPath_human, labelsFullPath_human)

print(resultimage)