import os
import cv2
import dlib
import imutils
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner, rect_to_bb


def create_graph(modelFullPath_human):
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.compat.v1.gfile.FastGFile(modelFullPath_human, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        _ = tf.import_graph_def(graph_def, name='')
    return graph


def preimg(input_file):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model_project/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=128)

    image = cv2.imread(input_file)
    # cv2.imshow('original', image)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    out_file = 'test.jpg'
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=128)
        faceAligned = fa.align(image, gray, rect)
        # cv2.imshow('after', faceAligned)
        cv2.waitKey(0)
        cv2.imwrite(out_file, faceAligned)
    return out_file


def run_inference_on_image_human(imagePath_human, modelFullPath_human, labelsFullPath_human):
    answer = None

    if not tf.compat.v1.gfile.Exists(imagePath_human):
        tf.logging.fatal('File does not exist %s', imagePath_human)
        return answer

    test_data = preimg(imagePath_human)
    image_data = tf.compat.v1.gfile.FastGFile(test_data, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    graph = create_graph(modelFullPath_human)

    with tf.compat.v1.Session(graph= graph) as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-3:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath_human, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n","") for w in lines]
        result = []
        for node_id in top_k:
            human_string = labels[node_id]
            human_string = (human_string.strip("' ")[:-2])
            human_string = human_string[2:]
            score = predictions[node_id]
            result_imagepath = 'media/resultimage/' + human_string + '.jpg'
            result_temp = {"score": score, "class": human_string, "path": result_imagepath}
            result.append(result_temp)
            # print('%s (score = %.5f)' % (human_string, score))
        answer = labels[top_k[0]]
        return result

# imagePath = './new.jpg'  # 웹캠으로 찍은 사진 테스트

