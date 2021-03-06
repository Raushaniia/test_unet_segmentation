import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_PATH = "input/stage1_train"
TEST_PATH = "input/stage1_test"

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
# Get train and test IDs
number_of_train_image = len([name for name in os.listdir(TRAIN_PATH + '/images') if os.path.isfile(os.path.join(TRAIN_PATH+ '/images', name))])
train_ids = list(range(1, number_of_train_image))
number_of_test_image = len([name for name in os.listdir(TEST_PATH + '/images') if os.path.isfile(os.path.join(TEST_PATH+ '/images', name))])
test_ids = list(range(1, number_of_test_image))

# Get and resize train images and masks
images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
labels = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH
    shape = imread(path + '/images/CHNCXR_000' + str(id_) + '_0' + '.png').shape
    img = imread(path + '/images/CHNCXR_000' + str(id_) + '_0' + '.png')[:, :]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    shape1 = img.shape
    images[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    labels[n] = mask

X_train = images
Y_train = labels

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH
    img = imread(path + '/images/' + 'CHNCXR_000'  + str(id_) + '_0'  + '.png')[:,:]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

def shuffle():
    global images, labels
    p = np.random.permutation(len(X_train))
    images = X_train[p]
    labels = Y_train[p]

def next_batch(batch_s, iters):
    if(iters == 0):
        shuffle()
    count = batch_s * iters
    return images[count:(count + batch_s)], labels[count:(count + batch_s)]

def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1

def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    tf.layers.batch_normalization(input_tensor, training=True)
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=tf.nn.relu, name=name)

X = tf.placeholder(tf.float32, [None, 128, 128, 1])
lr = tf.placeholder(tf.float32)

conv1 = conv2d(X, 32, 1, "Y0") #128

conv2 = conv2d(conv1, 64, 3, "Y2", strides=(2, 2)) #64

conv3 = conv2d(conv2, 128, 3,  "Y3", strides=(2, 2)) #32


deconv1 = deconv2d(conv3, 1, 32, 128, 128, "Y2_deconv") # 32
#skip1 = tf.concat([deconv1, conv3], 0)
#net = tf.nn.relu(skip1)
net = tf.nn.relu(deconv1)

deconv2 = deconv2d(net, 2, 64, 64, 128, "Y1_deconv", strides=[1, 2, 2, 1]) # 64
#skip2 = tf.concat([deconv2, conv2], 0)
#net = tf.nn.relu(skip2)
net = tf.nn.relu(deconv2)

deconv3 = deconv2d(net, 2, 128, 32, 64, "Y0_deconv", strides=[1, 2, 2, 1]) # 128
#skip3 = tf.concat([deconv3, conv1], 0)
#net = tf.nn.relu(skip3)
net = tf.nn.relu(deconv3)

logits = deconv2d(net, 1, 128, 1, 32, "logits_deconv") # 128

Y_ = tf.placeholder(tf.float32, [None, 128, 128, 1])
loss = tf.losses.sigmoid_cross_entropy(Y_, logits)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_count = 0
display_count = 1
number_images = 10
batch_size = 2
number_steps = int(number_images/batch_size)
for i in range(number_steps):
    # training on batches of 2 images with 2 mask images
    if (batch_count > 3):
        batch_count = 0

    batch_X, batch_Y = next_batch(batch_size, batch_count)
    batch_X = batch_X.reshape(batch_size, 128, 128, 1)
    batch_count += 1

    feed_dict = {X: batch_X, Y_: batch_Y, lr: 0.001}
    loss_value, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
    if (i % 500 == 0):
        print(str(display_count) + " training loss:", str(loss_value))
        display_count += 1

print("Done!")

ix = 4 #random
test_image = X_test[ix].astype(float)
imshow(test_image)
plt.show()

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

test_image = np.reshape(test_image, [-1, 128 , 128, 1])

test_data = {X:test_image}

test_mask = sess.run([logits],feed_dict=test_data)

test_mask = np.reshape(np.squeeze(test_mask), [IMG_WIDTH , IMG_WIDTH, 1])

for i in range(IMG_WIDTH):
    for j in range(IMG_HEIGHT):
            test_mask[i][j] = int(sigmoid(test_mask[i][j])*255)
imshow(test_mask.squeeze().astype(np.uint8))
plt.show()

