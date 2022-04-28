import os
import cv2
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import copy
# import Jetson.GPIO as GPIO


tf.disable_eager_execution()
# led_pin=[7,11]
verification_threshold = 100
led_state = False
image_size = 160

class FaceDetection:
    def __init__(self):
        self.load_face_detection()

    # OpenCV DNN Face Detector: the function takes the directory of the frozen .pb model and a .pbtxt file.
    def load_face_detection(self):
        model_path = "./Models/FaceDetection/opencv_face_detector_uint8.pb"
        model_pbtxt = "./Models/FaceDetection/opencv_face_detector.pbtxt"
        # Serialised model is loaded from disk
        self.net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)

    # Face detection in an image
    def detect_faces(self, image):
        height, width, channels = image.shape
        # The input image is loaded and an input blob is constructed for the image,
        # resizing it to a fixed value of 300x300 pixels and normalising it.
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        # The blob is passed through the network and the detections are obtained.
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        # Loop over detections
        for i in range(detections.shape[2]):
            # The confidence associated with the prediction is extracted.
            confidence = detections[0, 0, i, 2]
            # Weak detections are filtered out, checking that the confidence is higher than the threshold.
            if confidence > 0.5:
                # The coordinates delimiting the rectangle of the face are calculated.
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append([x1, y1, x2 - x1, y2 - y1])
        return faces


class FaceRecognition:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.net = self.load_face_recognition()
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    # Loading the face recognition model and initialising the tensors with default values.
    def load_face_recognition(self):
        model_path = "./Models/FaceRecognition/"
        saver = tf.train.import_meta_graph(os.path.join(model_path, "model-20180204-160909.meta"))
        saver.restore(self.session, os.path.join(model_path, "model-20180204-160909.ckpt-266000"))

    # Image to embedding conversion
    def img_to_embedding(self, img, image_size):
        # Creation of the image tensor
        image = np.zeros((1, image_size, image_size, 3))
        # Convert the image to rgb if it is in greyscale
        if img.ndim == 2:
            imagen = copy.deepcopy(img)
            w, h = imagen.shape
            img = np.empty((w, h, 3), dtype=np.uint8)
            img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = imagen
        # Pre - whitening to image
        std_adj = np.maximum(np.std(img), 1.0 / np.sqrt(img.size))
        img = np.multiply(np.subtract(img, np.mean(img)), 1 / std_adj)
        image[0, :, :, :] = img
        # Conversion to embedding
        feed_dict = {self.images_placeholder: image, self.phase_train_placeholder: False}
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.session.run(self.embeddings, feed_dict=feed_dict)
        return np.squeeze(emb_array)

    # Dataset image processing. Image transformation to 128-feature vector and saving in an embedding dictionary.
    def load_face_embeddings(self, image_dir, face_detector):
        embeddings = {}
        # Loop through all images in the database
        for file in os.listdir(image_dir):
            image = cv2.imread(image_dir + file)
            embeddings[file.split(".")[0]] = self.img_to_embedding(cv2.resize(image, (160, 160)), image_size)
        return embeddings

    @staticmethod
    # Compares two matrices and returns the difference between them as a scalar value.
    def is_same(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff

# Face recognition
def face_recognition(image, embeddings, face_detector, face_recognizer, show=False):
    # Transformation to embedding
    user_embed = face_recognizer.img_to_embedding(cv2.resize(image, (160, 160)), image_size)
    detected = {}
    # Loop of all embeddings in the dataset
    for _user in embeddings:
        # Comparison of the embedding of the unknown image with the one in the dataset
        thresh = face_recognizer.is_same(embeddings[_user], user_embed)
        # Faces with a distance less than the verification_threshold are saved.
        detected[_user] = thresh
    # Sorting from least to most distance of saved embeddings
    detected = {k: v for k, v in sorted(detected.items(), key=lambda item: item[1])}
    detected = list(detected.keys())

    return detected


# def initialise_led():
# GPIO.setmode(GPIO.board)
# GPIO.setup(led_pin, GPIO.OUT)
# # Red LED on and green LED off
# GPIO.output(led_pin, (GPIO.HIGH, GPIO.HIGH))

# def change_led(green_state):
#     if (led_state and green_state):
#         # Green LED off
#         GPIO.output(led_pin, (GPIO.HIGH, GPIO.HIGH))
#         led_state=False
#     else:
#         # Green LED on
#         GPIO.output(led_pin, (GPIO.LOW, GPIO.LOW))
#         led_state=True


# Function to be called in the main
def main_program(image_or_video_path=None, show=False, dataset="./FUERA/"):
    fd = FaceDetection()
    fr = FaceRecognition()
    # Dataset embeddings
    embeddings = fr.load_face_embeddings(dataset, fd)
    waitkey_variable = 1
    image_flip = False
    # If input is an image or video
    if image_or_video_path:
        print("Using path: ", image_or_video_path)
        image = cv2.imread(image_or_video_path)
        print(face_recognition(image, embeddings, fd, fr, show))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Path to input file')
    parser.add_argument("--show", action="store_true", help="Show mage or video")
    parser.add_argument('--dataset', type=str, default="./FUERA/", help='Path to dataset')
    args = parser.parse_args()

    main_program(args.input, args.show, args.dataset)
