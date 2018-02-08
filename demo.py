import numpy as np
import tensorflow as tf
import cv2
import os
from resnet18_linknet import LinkNet_resnt18


def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color


root = "/home/thinkjoy/dataset/"
data_dir = root + "Cityscapes"
project_dir = './logs/'

batch_size = 4
img_height = 512
img_width = 1024
num_classes = 20

inputs = tf.placeholder(dtype=tf.float32, shape=(None, img_height, img_width, 3), name='input_img')
LinkNet_resnt18_model = LinkNet_resnt18(inputs, num_classes=num_classes, is_training=False)
logits, _ = LinkNet_resnt18_model.build_model()
pred = tf.nn.softmax(logits, name='logits_to_softmax')

# load the sequence data:
seq_frames_dir = data_dir + "/leftImg8bit/demoVideo/stuttgart_02/"
seq_frame_paths = []
frame_names = sorted(os.listdir(seq_frames_dir))
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print(step)

    frame_path = seq_frames_dir + frame_name
    seq_frame_paths.append(frame_path)

# compute the number of batches needed to iterate through the data:
no_of_frames = len(seq_frame_paths)
no_of_batches = int(no_of_frames/batch_size)

# define where to place the resulting images:
results_dir = project_dir + "results_on_seq/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# create a saver for restoring variables/parameters:
saver = tf.train.Saver()

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    # restore the best trained model:
    checkpoints = project_dir + "model/checkpoints"
    if os.path.isfile(checkpoints):
        checkpoints_path = checkpoints
    else:
        checkpoints_path = tf.train.latest_checkpoint(checkpoints)
    saver.restore(sess, checkpoints_path)

    batch_pointer = 0
    for step in range(no_of_batches):
        nor_batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        img_paths = []

        for i in range(batch_size):
            img_path = seq_frame_paths[batch_pointer + i]
            img_paths.append(img_path)

            # read the image:
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (img_width, img_height))
            batch_imgs[i] = img

            nor_img = img.astype(np.float32) / 255.0
            nor_batch_imgs[i] = nor_img

        batch_pointer += batch_size

        feed_dict = {inputs: nor_batch_imgs}

        # run a forward pass and get the logits:
        p = sess.run(pred, feed_dict=feed_dict)

        print("step: %d/%d" % (step+1, no_of_batches))

        # save all predicted label images overlayed on the input frames to results_dir:
        predictions = np.argmax(p, axis=3)
        for i in range(batch_size):
            pred_img = predictions[i]
            pred_img_color = label_img_to_color(pred_img)

            img = batch_imgs[i]

            img_file_name = img_paths[i].split("/")[-1]
            img_name = img_file_name.split(".png")[0]
            pred_path = results_dir + img_name + "_pred.png"

            overlayed_img = 0.3*img + 0.7*pred_img_color

            cv2.imwrite(pred_path, overlayed_img)

# create a video of all the resulting overlayed images:
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
out = cv2.VideoWriter(results_dir + "cityscapes_stuttgart_02_pred.avi", fourcc, 20.0, (img_width, img_height))

frame_names = sorted(os.listdir(results_dir))[:300]
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print(step)

    if ".png" in frame_name:
        frame_path = results_dir + frame_name
        frame = cv2.imread(frame_path, -1)

        out.write(frame)