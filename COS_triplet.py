#python news.py --data_dir=data --batch_size=1 --mode=cmc
#python news.py --mode=test --image1=data/labeled/val/0046_00.jpg --image2=data/labeled/val/0049_07.jpg
import tensorflow as tf
import numpy as np
import cv2

#import cuhk03_dataset_label2
import big_dataset_label as cuhk03_dataset_label2
import time

import random
import cmc

import tensorflow.contrib.slim as slim
import nets.deep_sort.network_definition as net

import losses

from triplet_loss import batch_hard_triplet_loss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt  
from PIL import Image 

print tf.__version__
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '128', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '330000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs_triplet2/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '1e-3', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

tf.flags.DEFINE_float('global_rate', '1.0', 'global rate')
tf.flags.DEFINE_float('local_rate', '1.0', 'local rate')
tf.flags.DEFINE_float('softmax_rate', '1.0', 'softmax rate')

tf.flags.DEFINE_integer('ID_num', '32', 'id number')
tf.flags.DEFINE_integer('IMG_PER_ID', '4', 'img per id')

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 128




def preprocess(images, is_train):
    def train():    
        split = tf.split(images, [1, 1,1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                #split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT , IMAGE_WIDTH , 3])
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[2], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1,1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)


def _create_softmax_loss(feature_var, logit_var, label_var):
    del feature_var  # Unused variable
    cross_entropy_var = slim.losses.sparse_softmax_cross_entropy(
        logit_var, tf.cast(label_var, tf.int64))
    tf.summary.scalar("cross_entropy_loss", cross_entropy_var)

    accuracy_var = slim.metrics.accuracy(
        tf.cast(tf.argmax(logit_var, 1), tf.int64), label_var)
    tf.summary.scalar("classification accuracy", accuracy_var)


def _create_magnet_loss(feature_var, logit_var, label_var, monitor_mode=False):
    del logit_var  # Unusued variable
    magnet_loss, _, _ = losses.magnet_loss(feature_var, label_var)
    tf.summary.scalar("magnet_loss", magnet_loss)
    if not monitor_mode:
        slim.losses.add_loss(magnet_loss)


def _create_triplet_loss(feature_var, logit_var, label_var, monitor_mode=False):
    del logit_var  # Unusued variables
    triplet_loss = losses.softmargin_triplet_loss(feature_var, label_var)
    tf.summary.scalar("triplet_loss", triplet_loss)
    if not monitor_mode:
        slim.losses.add_loss(triplet_loss)

def _create_loss(
        feature_var, logit_var, label_var, mode, monitor_magnet=True,
        monitor_triplet=True):
    if mode == "cosine-softmax":
        _create_softmax_loss(feature_var, logit_var, label_var)
    elif mode == "magnet":
        _create_magnet_loss(feature_var, logit_var, label_var)
    elif mode == "triplet":
        _create_triplet_loss(feature_var, logit_var, label_var)
    else:
        raise ValueError("Unknown loss mode: '%s'" % mode)

    if monitor_magnet and mode != "magnet":
        _create_magnet_loss(
            feature_var, logit_var, label_var, monitor_mode=monitor_magnet)
    if monitor_triplet and mode != "triplet":
        _create_triplet_loss(
            feature_var, logit_var, label_var, monitor_mode=True)
       


def main(argv=None):
    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1
    
    if FLAGS.mode == 'cmc':
        FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    #images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    images = tf.placeholder(tf.float32, [3, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    
    images_total = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images_total')
    
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size], name='labels')
   

    
    label_var = tf.placeholder(tf.int64, (None,))
    
    

    
    
    
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    if FLAGS.mode == 'train':
        tarin_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'train')
        print(tarin_num_id, '               11111111111111111111               1111111111111111')
    elif FLAGS.mode == 'val':
        val_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'val')
    #images1, images2,images3 = preprocess(images, is_train)
    
    
    
    
    img_temp = []
    for t in range(FLAGS.batch_size):
        
        
        images_total_ex = tf.image.random_flip_left_right(images_total[t])
        images_total_ex = tf.image.random_brightness(images_total_ex, max_delta=32. / 255.)
        img_temp.append(images_total_ex)
    images_total = tf.cast(img_temp,tf.float32)
    
    
    
    network_factory = net.create_network_factory(
            is_training=True, num_classes=1500 + 1,
            add_logits= "triplet")
            
    feature_var, logit_var = network_factory(images_total)
    _create_loss(feature_var, logit_var, label_var, mode="triplet")
   
    
    
    
    trainable_scopes=None
    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        variables_to_train = []
        for scope in trainable_scopes:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
    

    #global_step = tf.train.get_or_create_global_step()

    loss_var = tf.losses.get_total_loss()
    train_op = slim.learning.create_train_op(
        loss_var, tf.train.AdamOptimizer(learning_rate=learning_rate),
        global_step, summarize_gradients=False,
        variables_to_train=variables_to_train)
    
    
    tf.summary.scalar("total_loss", loss_var)
    tf.summary.scalar("learning_rate", learning_rate)

    regularization_var = tf.reduce_sum(tf.losses.get_regularization_loss())
    tf.summary.scalar("weight_loss", regularization_var)
    

    #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    #train = optimizer.minimize(loss, global_step=global_step)
    
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    #train = optimizer.minimize(loss, global_step=global_step)
    
   

    lr = FLAGS.learning_rate

    #config=tf.ConfigProto(log_device_placement=True)
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) 
    # GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    
    with tf.Session(config=config) as sess:
        
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)

        
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)


        
        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in xrange(step, FLAGS.max_steps + 1):

                batch_images, batch_labels, batch_images_total = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)

              
                feed_dict = {learning_rate: lr,  is_train: True , images_total: batch_images_total, labels: batch_labels, label_var : batch_labels}
               
                
                
                
                start_time = time.time()
                #_,train_loss = sess.run([train,loss], feed_dict=feed_dict) 
                _,train_loss = sess.run([train_op,loss_var], feed_dict=feed_dict) 
                duration = time.time() - start_time
                train_loss = np.array(train_loss)
                print   duration,'duration'  
                print('Step: %d, Learning rate: %f, Train loss: %f ' % (i, lr, train_loss))

                
                
                if i % 20 == 0:
                    result = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(result, i)
          
                
          
                
                
                lr = FLAGS.learning_rate * ((0.1) ** (i/180000)) 
                #lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 1000 == 0:
                    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)
                    # test save
                    #vgg.save_npy(sess, './big.npy')            
                
                
                
                
        

        elif FLAGS.mode == 'val':
            total = 0.
            for _ in xrange(10):
                batch_images, batch_labels = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'val', val_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {images: batch_images, labels: batch_labels, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                prediction = np.argmax(prediction, axis=1)
                label = np.argmax(batch_labels, axis=1)

                for i in xrange(len(prediction)):
                    if prediction[i] == label[i]:
                        total += 1
            print('Accuracy: %f' % (total / (FLAGS.batch_size * 10)))

            '''
            for i in xrange(len(prediction)):
                print('Prediction: %s, Label: %s' % (prediction[i] == 0, labels[i] == 0))
                image1 = cv2.cvtColor(batch_images[0][i], cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(batch_images[1][i], cv2.COLOR_RGB2BGR)
                image = np.concatenate((image1, image2), axis=1)
                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == 1048603:  # ESC key
                    break
            '''


        
        
        
        elif FLAGS.mode == 'test':
            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            test_images = np.array([image1, image2,image2])

            feed_dict = {images: test_images, is_train: False, droup_is_training: False}
            #prediction, prediction2 = sess.run([DD,DD2], feed_dict=feed_dict)
            prediction = sess.run([inference], feed_dict=feed_dict)
            prediction = np.array(prediction)
            print prediction.shape
            print( np.argmax(prediction[0])+1)
            
           
        
            #print(bool(not np.argmax(prediction[0])))

if __name__ == '__main__':
    tf.app.run()