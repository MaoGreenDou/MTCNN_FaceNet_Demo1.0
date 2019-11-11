#-*- coding: utf-8 -*-
__author__ = 'MaoDou'
__date__ = '2019/11/10 20:48'

from scipy import misc
import tensorflow as tf
import detect_face
import cv2
import matplotlib.pyplot as plt

import os           #---------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     #---------------------

# %pylab inline
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)     #---------------------
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))     #---------------------

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

image_path = 'D.jpg'

img = misc.imread(image_path)
bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]  # 人脸数目
print('找到人脸数目为：{}'.format(nrof_faces))

for face_position in bounding_boxes:
    face_position = face_position.astype(int)

    # print(face_position[0:4])

    cv2.rectangle(img, (face_position[0] - 10, face_position[1] - 10), (face_position[2] + 10, face_position[3] + 10),
                  (0, 255, 0), 5)

plt.imshow(img)
plt.show()
