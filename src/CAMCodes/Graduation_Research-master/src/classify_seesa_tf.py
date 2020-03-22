
import tensorflow as tf
from sklearn import cross_validation
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

STANDARD_SIZE = (300,250)#(300, 167)

# Reads an image from a file. decodes it into a dense tensor, and resizes it to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [250, 250])
    return image_resized,  label


def input_data():

    with open("path_and_label.txt", "r") as f:
        train_path_list = f.readlines()

    filenames = []
    labels = []
    for row in train_path_list:
        row = row.split(" ")
        filenames.append(row[0])
        labels.append(row[1])

    label = []
    for i in labels:
        label.append(int(i))

    train_data, test_data, train_target, test_target = cross_validation.train_test_split(filenames, label, test_size=0.3, random_state=0)
    return train_data, test_data, train_target, test_target


def inference(x):
    init = tf.constant_initializer(value=0)
    x = tf.placeholder("tf.float32", [569, 187500])
    w = tf.get_variable("w", [62500, 2], initializer=init)
    b = tf.get_variable("b", [2], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, w)+b)
    return output


def loss(output, y):
    elementwise_product = y * tf.log(output)
    xentropy = -tf.reduce_sum(elementwise_product, reduction_indices=1)
    loss = tf.reduce_mean(xentropy)
    return loss


# 画像をパース
def img_to_matrix(filename, verbose=False):
    img = Image.open(filename)
    if verbose:
        print ('changing size from %s to %s' % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    imgArray = np.asarray(img)
    return imgArray  # imgArray.shape = (167 x 300 x 3)


#1次元に引き延ばす
def flatten_image(img):
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]



def main():
    train_data, test_data, train_label, test_label, = input_data()

    print(train_data)
    images = train_data
    labels = train_label
    ls = []

    for i in labels:
        if i == 1:
            ls.append("cloudy_seesaa")
        elif i == 0:
            ls.append("sunny_seesaa")
    labels = ls

    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)

    is_train = np.random.uniform(0, 1, len(data)) <= 0.7
    y = np.where(np.array(labels) == 'cloudy_seesaa', 1, 0)

    train_x, train_y = data[is_train], y[is_train]

    # plot in 2 dimensions
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(data)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1],
                       "label": np.where(y == 1, 'cloudy_seesaa', 'sunny_seesaa')})
    colors = ['blue', 'red']

    plt.figure(figsize=(10, 10))
    for label, color in zip(df['label'].unique(), colors):
        mask = df['label'] == label
        plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
    sns.set()
    plt.xlabel("pc1") #全データの分散が最大となる方向
    plt.ylabel("pc2") #第一主成分に垂直な方向の軸
    plt.legend()
    plt.show()
    # plt.savefig('pca_feature1.png')

    # training a classifier
    pca = RandomizedPCA(n_components=5)
    train_x = pca.fit_transform(train_x)

    svm = LinearSVC(C=1.0)
    svm.fit(train_x, train_y)
    joblib.dump(svm, 'model.pkl')

    # evaluating the model
    test_x, test_y = data[is_train == False], y[is_train == False]
    test_x = pca.transform(test_x)
    print(pd.crosstab(test_y, svm.predict(test_x),
                      rownames=['Actual'], colnames=['Predicted']))

    # A vector of filenames.

    train_filenames = tf.constant(train_data)
    test_filenames = tf.constant(test_data)
    # labels[i] is the label for the image in filenames[i]
    train_labels = tf.constant(train_label)
    test_labels = tf.constant(test_label)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))

    train_dataset = train_dataset.map(_parse_function)#.batch(4)
    test_dataset = test_dataset.map(_parse_function)#.batch(4)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_next_element, train_lable = iterator.get_next()
    test_next_element, test_label = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    sess = tf.InteractiveSession()
    sess.run(training_init_op)
    sess.run(test_init_op)

    for i in range(4):
        consequence = tf.convert_to_tensor(train_next_element)
        print(type(consequence.eval()))
    for i in range(4):
        print(train_lable.eval())
    sess.close()


if __name__ == '__main__':
    main()


