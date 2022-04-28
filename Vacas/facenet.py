import os
import cv2
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import copy

tf.disable_eager_execution()
image_size = 160

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
    def load_face_embeddings(self, image_dir):
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
def face_recognition(image, embeddings, face_recognizer):
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

# Function to be called in the main
def main_program(image_or_video_path=None, show=False, dataset="./FUERA/"):
    fr = FaceRecognition()
    # Dataset embeddings
    embeddings = fr.load_face_embeddings(dataset)

    for i in os.listdir(image_or_video_path):
        print("Using path: ", i)
        for j in os.listdir(image_or_video_path + i):
            image = cv2.imread(image_or_video_path + i + '/' +j)
            print(face_recognition(image, embeddings, fr))
        print('-----------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Path to input file')
    parser.add_argument("--show", action="store_true", help="Show mage or video")
    parser.add_argument('--dataset', type=str, default="./FUERA/", help='Path to dataset')
    args = parser.parse_args()
    main_program(args.input, args.show, args.dataset)
