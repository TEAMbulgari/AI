import numpy as np
import tensorflow as tf




def create_graph(modelFullPath):
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.compat.v1.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        _ = tf.import_graph_def(graph_def, name='')
    return graph


def run_inference_on_image(imagePath, modelFullPath, labelsFullPath):
    answer = None

    if not tf.compat.v1.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.compat.v1.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    graph = create_graph(modelFullPath)

    with tf.compat.v1.Session(graph= graph) as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-3:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        result = []
        for node_id in top_k:
            animal_string = labels[node_id]
            animal_string = (animal_string.strip("' ")[:-2])
            animal_string = animal_string[2:]
            score = predictions[node_id]
            result_imagepath = './resultimage_animal/' + animal_string + '.jpg'
            result_temp = {"score": score, "class": animal_string, "path": result_imagepath}
            result.append(result_temp)
            # print('%s (score = %.5f)' % (animal_string, score))
        answer = labels[top_k[0]]
        return result


