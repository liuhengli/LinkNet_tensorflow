# -*- coding: utf-8 -*-
import tensorflow as tf
import time
from load_cityscapes_data import SegmentedData
from resnet18_linknet import LinkNet_resnt18
import datetime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
-------------------------------------------------
   File Name：     train
   version:        v1.0 
   Description :
   Author :       liuhengli
   date：          18-1-11
   license:        Apache Licence
-------------------------------------------------
   Change Activity:
                   18-1-11:
-------------------------------------------------
"""
__author__ = 'liuhengli'


class Train_LinkNet:
    def __init__(self, img_height=512,
                 img_width=1024,
                 hm_shape=(64, 64),
                 num_classes=20,
                 batch_size=4,
                 learn_rate=5e-4,
                 decay=0.96,
                 decay_step=2000,
                 training=True,
                 logdir_train=None,
                 logdir_test=None,
                 save_path=None,
                 name='LinkNet'):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.batchSize = batch_size
        self.training = training
        self.lr_values = [2.5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
        self.learning_rate = learn_rate
        self.decay = decay
        self.decay_step = decay_step
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.save_path = save_path
        self.name = name

        self.end_learning_rate = self.learning_rate / 100
        self.learning_rate_decay_factor = 0.96
        self.learning_rate_decay_type = 'exponential'
        self.optimizer_name = 'rmsprop'

        self.adadelta_rho = 0.95
        self.adagrad_initial_accumulator_value = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.ftrl_learning_rate_power = -0.5
        self.ftrl_initial_accumulator_value = 0.1
        self.ftrl_l1 = 0.0
        self.ftrl_l2 = 0.0
        self.momentum = 0.9

        self.rmsprop_decay = 0.9
        self.rmsprop_momentum = 0.9
        self.opt_epsilon = 1e-8

    def __create_model_dirs(self):
        self.model_dir = self.save_path
        self.checkpoints_dir = self.model_dir + "checkpoints/"
        self.debug_imgs_dir = self.model_dir + "imgs/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.debug_imgs_dir)

    def __configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.

        Args:
            learning_rate: A scalar or `Tensor` learning rate.

        Returns:
            An instance of an optimizer.

        Raises:
            ValueError: if FLAGS.optimizer is not recognized.
        """
        if self.optimizer_name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self.adadelta_rho,
                epsilon=self.opt_epsilon)
        elif self.optimizer_name == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=self.adagrad_initial_accumulator_value)
        elif self.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=self.adam_beta1,
                beta2=self.adam_beta2,
                epsilon=self.opt_epsilon)
        elif self.optimizer_name == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self.ftrl_learning_rate_power,
                initial_accumulator_value=self.ftrl_initial_accumulator_value,
                l1_regularization_strength=self.ftrl_l1,
                l2_regularization_strength=self.ftrl_l2)
        elif self.optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self.momentum,
                name='Momentum')
        elif self.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self.rmsprop_decay,
                momentum=self.rmsprop_momentum,
                epsilon=self.opt_epsilon)
        elif self.optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.optimizer)
        return optimizer

    def MSE_loss(self, logits, labels):
        mse = tf.losses.mean_squared_error(labels, logits, scope='mes_loss')
        return mse

    def generate_model(self, optimizer):
        """ Create the complete graph
        """
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(dtype=tf.float32, shape=(None, self.img_height, self.img_width, 3), name='input_img')
            self.gtMaps = tf.placeholder(dtype=tf.int32, shape=(None, self.img_height, self.img_width, 1))
        # TODO : Implement weighted loss function
        # NOT USABLE AT THE MOMENT
        inputTime = time.time()
        print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
        # build model
        self.LinkNet_resnt18_model = LinkNet_resnt18(self.img, num_classes=self.num_classes, is_training=True)
        self.logits, _ = self.LinkNet_resnt18_model.build_model()
        graphTime = time.time()
        print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')

        # compute the weighted cross-entropy segmentation loss for each pixel:
        # print(self.gtMaps.shape, self.logits.shape)
        with tf.name_scope('seg_loss'):
            self.seg_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.gtMaps, logits=self.logits)
        with tf.name_scope('regularization_loss'):
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
        with tf.name_scope('total_loss'):
            self.loss = self.regularization_loss + self.seg_loss
        lossTime = time.time()
        print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')

        with tf.name_scope('steps'):
            self.train_step = tf.Variable(0, trainable=False, name='global_step')

        lrTime = time.time()
        # print('---LR : Done (' + str(int(abs(accurTime - lrTime))) + ' sec.)')
        # with tf.device(self.gpu):
        if optimizer == 'rmsprop':
            print("Using RMSPropOptimizer ...")
            with tf.name_scope('lr'):
                epochSize = 2000
                LR_EPOCH = [50, 90, 120, 200]
                boundaries = [int(epoch * epochSize) for epoch in LR_EPOCH]
                self.lr = tf.train.piecewise_constant(self.train_step, boundaries, self.lr_values,
                                                      name='fixed_learning_rate')
            with tf.name_scope('rmsprop'):
                self.trainop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if optimizer == 'adam':
            print("Using AdamOptimizer ...")
            with tf.name_scope('lr'):
                self.lr = self.learning_rate
            with tf.name_scope('adam'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        optimTime = time.time()
        print('---Optim : Done (' + str(int(abs(optimTime - lrTime))) + ' sec.)')
        with tf.name_scope('minimizer'):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.train_step)
        minimTime = time.time()
        print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')
        initTime = time.time()
        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')

        # summary
        with tf.name_scope('training'):
            tf.summary.scalar('seg_loss', self.seg_loss, collections=['train'])
            tf.summary.scalar('regularization_loss', self.regularization_loss, collections=['train'])
            tf.summary.scalar('loss', self.loss, collections=['train'])
            # tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        with tf.name_scope('Valid'):
            tf.summary.scalar('valid_loss', self.seg_loss, collections=['test'])
        # with tf.name_scope('summary'):
        #     for i in range(self.outDim):
        #         tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])
        self.train_sumamary_op = tf.summary.merge_all('train')
        self.test_sumamary_op = tf.summary.merge_all('test')
        endTime = time.time()
        print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')
        del endTime, startTime, initTime, optimTime, minimTime, lrTime, lossTime, graphTime, inputTime

    def restore(self, load=None, restore_variables=None):
        """ Restore a pretrained model
        Args:
            load	: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            # with tf.device(self.gpu):
            # self._init_session()
            self._define_saver_summary(summary=True, restore_variables=restore_variables)
            if load is not None:
                print('Loading Trained Model')
                t = time.time()
                self.saver.restore(self.Session, load)
                print('Model Loaded (', time.time() - t, ' sec.)')
            else:
                print('No Model load ...')

    def _define_saver_summary(self, summary=True, restore_variables=None):
        """ Create Summary and Saver
        Args:
            logdir_train		: Path to train summary directory
            logdir_test		: Path to test summary directory
        """
        if (self.logdir_train == None) or (self.logdir_test == None):
            raise ValueError('Train/Test directory not assigned')
        else:
            self.saver = tf.train.Saver(var_list=restore_variables)
            if summary:
                self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                self.test_summary = tf.summary.FileWriter(self.logdir_test)

    def _train(self, nEpochs=200, epochSize=2000, saveStep=100, validIter=10):
        """
        """
        val_loss_per_epoch = []
        with tf.name_scope('Train'):
            # train_dataset = LS3D(self.train_dir, 'train_dataset', image_size=64)
            # val_dataset = LS3D(self.val_dir, 'val_dataset', image_size=64)
            best_epoch_losses = [1000, 1000, 1000, 1000, 1000]
            saver = tf.train.Saver(max_to_keep=50)
            for epoch in range(nEpochs):
                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                # Training Set
                for i in range(epochSize):
                    img_train, gt_train = self.train_dataset.get_batch(self.batchSize)
                    feed_dict = {self.img: img_train, self.gtMaps: gt_train}
                    if (i+1) % saveStep == 0:
                        _, loss, summary, n_step = self.Session.run([self.train_op, self.loss, self.train_sumamary_op, self.train_step], feed_dict=feed_dict)
                        self.train_summary.add_summary(summary, n_step)
                        self.train_summary.flush()
                        print("%s : Epoch: %d, Step: %d, loss: %6f" % (datetime.datetime.now(), epoch, n_step, loss))
                    else:
                        # print("=====", img_train.shape)
                        _, n_step = self.Session.run([self.train_op, self.train_step], feed_dict=feed_dict)

                # Validation Set
                val_loss = self._evaluate_on_val(epoch, n_step)
                # save the val epoch loss:
                val_loss_per_epoch.append(val_loss)
                # (if top 5 performance on val:)
                if val_loss < max(best_epoch_losses): 
                    # save the model weights to disk:
                    saver.save(self.Session, self.checkpoints_dir, n_step)
                    print("checkpoint saved in file: %s" % self.checkpoint_path)

                    # update the top 5 val losses:
                    index = best_epoch_losses.index(max(best_epoch_losses))
                    best_epoch_losses[index] = val_loss

                # plot the val loss vs epoch and save to disk:
                plt.figure(1)
                plt.plot(val_loss_per_epoch, "k^")
                plt.plot(val_loss_per_epoch, "k")
                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.title("validation loss per epoch")
                plt.savefig("%sval_loss_per_epoch.png" % self.save_path)
                plt.close(1)

    def _evaluate_on_val(self, epoch, n_step):
        val_batch_losses = []
        # batch_pointer = 0
        no_of_val_batches = int(self.no_of_val_imgs/self.batchSize)
        for step in range(no_of_val_batches):
            img_valid, gt_valid = self.val_dataset.get_batch(self.batchSize)
            # gt_valid = np.asarray(gt_valid)
            # gt_valid = np.reshape(gt_valid, (self.batchSize, self.img_height, self.img_width, 1))        
            feed_dict = {self.img: img_valid, self.gtMaps: gt_valid}
            # if step == (no_of_val_batches - 1):
            #     batch_loss, logits, summary = self.Session.run([self.loss, self.logits, self.test_sumamary_op], feed_dict=feed_dict)
            #     val_batch_losses.append(batch_loss)
            #     with tf.name_scope('val_loss'):
            #         val_loss = np.mean(val_batch_losses)
            #         self.valid_loss = val_loss
            #     self.test_summary.add_summary(summary, n_step)
            #     self.test_summary.flush()
            #     print("validation loss: %g" % val_loss)
            # else:
            #     # run a forward pass, get the batch loss and the logits:
            batch_loss, logits = self.Session.run([self.loss, self.logits], feed_dict=feed_dict)
            val_batch_losses.append(batch_loss)
            if (step+1) % 20 == 0:
                print("epoch: %d/%d, val step: %d/%d, val batch loss: %g" % (epoch + 1,  self.nEpochs, step + 1, no_of_val_batches, batch_loss))

            if step < 4:
                # save the predicted label images to disk for debugging and
                # qualitative evaluation:
                predictions = np.argmax(logits, axis=3)
                for i in range(self.batchSize):
                    pred_img = predictions[i]
                    label_img_color = self.label_img_to_color(pred_img)
                    cv2.imwrite((self.debug_imgs_dir + "val_" + str(epoch) + "_" +
                                 str(step) + "_" + str(i) + ".png"), label_img_color)
        val_loss = np.mean(val_batch_losses)
        print("validation loss: %g" % val_loss)
        return val_loss

    # function for colorizing a label image:
    def label_img_to_color(self, img):
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
            13: [  0,  0,255],
            14: [  0, 255, 0],
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

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.Session = tf.Session(config=config)
        t_start = time.time()
        self.init = tf.global_variables_initializer()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def run(self):
        self.root = "/home/thinkjoy/dataset/"
        self.data_dir = self.root + "Cityscapes"

        self.logdir_train = './logs/train'
        self.logdir_test = './logs/test'
        self.save_path = './logs/model/'

        self.nEpochs = 300
        self.batchSize = 8
        # self.learning_rate = 5e-4
        self.learning_rate = 5e-5
        self.saveStep = 20
        self.checkpoint_path = None
        # checkpoint_path = '/home/sdb/project/Hourglass_1222_s_56_1_sigma4/model'

        self.learning_rate_decay_type = 'fixed'
        self.optimizer_name = 'adam'

        # load data
        self.train_dataset = SegmentedData(self.data_dir, 'train', new_img_height=self.img_height, new_img_width=self.img_width)
        self.val_dataset = SegmentedData(self.data_dir, 'val', new_img_height=self.img_height, new_img_width=self.img_width)
        self.no_of_val_imgs = self.val_dataset._num_examples
        self.no_of_train_imgs = self.train_dataset._num_examples
        self.epochSize = self.no_of_train_imgs // self.batchSize

        # build model
        self.generate_model(self.optimizer_name)

        if tf.train.latest_checkpoint(self.save_path):
            load = tf.train.latest_checkpoint(self.save_path)
            print(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % load)
            variables = tf.all_variables()
        else:
            if self.checkpoint_path is not None:
                load = tf.train.latest_checkpoint(self.checkpoint_path)
                variables = tf.trainable_variables()
                print('Fine-tuning from %s' % load)
            else:
                load = None
                variables = tf.all_variables()

        with tf.name_scope('Session'):
            self._init_weight()
            self.__create_model_dirs()
            # print(variables)
            self.restore(load, variables)
            self._train(self.nEpochs, self.epochSize, self.saveStep, validIter=10)


if __name__ == '__main__':
    obj = Train_LinkNet()
    obj.run()