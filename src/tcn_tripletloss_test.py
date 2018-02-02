from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, merge, Flatten, Embedding, Dense, GlobalAveragePooling2D, Lambda
from keras import backend as K
import keras
from keras.models import load_model, model_from_json
import random
import keras.backend.tensorflow_backend as KTF

import numpy as np
from glob import glob
from os import path
import os
import sys

import pandas as pd
import logging

# argv[]
# argv[1] : input directory for test_anchor
# argv[2] : input directory for test_positive
# argv[3] : epoch_num(max)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def triplet_loss(vec, alpha = 0.2):
    anchor, positive, negative = vec
    d_p = K.sum(K.square(anchor - positive), axis = -1)
    d_n = K.sum(K.square(anchor - negative), axis = -1)
    loss = K.mean(K.maximum(0.0, d_p -d_n + alpha ))
    return loss

def create_base_network(model_name):
    model_name = 'inception'
    if model_name == 'vgg16':
        vgg_model = VGG16(include_top=False, weights='imagenet')
        model = Model(input = vgg_model.input, output = vgg_model.output)
    elif model_name == 'inception':
        inception_model = InceptionV3(weights='imagenet', include_top=False)
        x = inception_model.get_layer('mixed5').output
        x = GlobalAveragePooling2D()(x)


        x = Dense(1024, activation='relu')(x)
        x = Dense(128)(x)
        model = Model(input = inception_model.input, output = x)
    return model

def triplet_output_shape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)

def build(base_model, input_shape=(224, 224,3)):
    anchor = Input(shape=input_shape, name='input_anchor')
    positive = Input(shape=input_shape, name='input_positive')
    negative = Input(shape=input_shape, name='input_negative')
    out_a = base_model(anchor)
    out_p = base_model(positive)
    out_n = base_model(negative)

    loss = Lambda(triplet_loss,
                  output_shape=triplet_output_shape,
                  name='triplet_loss')([out_a, out_p, out_n])

    return Model(input=[anchor, positive, negative], output=loss)

def build_predict(base_model, input_shape=(224, 224,3)):
    input_img = Input(shape=input_shape)
    out = base_model(input_img)
    return Model(input=input_img, output=out)


def get_img(name):
    img = image.load_img(name, target_size=(224, 224))
    x = image.img_to_array(img)
    return x

def batch_generator(pairs, num_batches = 32):
    while True:
        anc =[]
        pos = []
        neg = []
        for i, pair in enumerate(pairs):
            pos.append(get_img(pair['positive']))
            anc.append(get_img(pair['anchor']))
            neg.append(get_img(pair['negative']))

            if len(anc) == num_batches:
                yield({'input_positive':np.asarray(pos, dtype=np.float32), 'input_negative':np.asarray(neg, dtype=np.float32), 'input_anchor':np.asarray(anc, dtype=np.float32)},{'triplet_loss':np.zeros((num_batches, 1), dtype = np.float32)})
                pos = []
                anc = []
                neg = []


def random_batch_generator(train_list, num_batches = 32):
    while True:
        anc =[]
        pos = []
        neg = []
        np.random.shuffle(train_list)
        for i in range(len(train_list)):
            frame = random.randint(0, len(train_list[i])-1)
            anchor = train_list[i][frame][0]
            positive = train_list[i][frame][1]
            while True:
               n = random.randint(0, len(train_list[i])-1)
               if n != frame:
                   break
            negative = train_list[i][n][1]
            pos.append(get_img(positive))
            anc.append(get_img(anchor))
            neg.append(get_img(negative))

            if len(anc) == num_batches:
                yield({'input_positive':np.asarray(pos, dtype=np.float32), 'input_negative':np.asarray(neg, dtype=np.float32), 'input_anchor':np.asarray(anc, dtype=np.float32)},{'triplet_loss':np.zeros((num_batches, 1), dtype = np.float32)})
                pos = []
                anc = []
                neg = []

def test():
    # ログの出力名を設定（1）
    logger = logging.getLogger('LoggingTest')

    # ログレベルの設定（2）
    logger.setLevel(20)
    # ログレベル以下のログは標準出力に表示されない
    # NOTSET	0	設定値などの記録（全ての記録）
    # DEBUG	10	動作確認などデバッグの記録
    # INFO	20	正常動作の記録
    # WARNING	30	ログの定義名
    # ERROR	40	エラーなど重大な問題
    # CRITICAL	50	停止など致命的な問題

    # ログのファイル出力先を設定（3）
    fh = logging.FileHandler('output_test.log')
    logger.addHandler(fh)

    # ログのコンソール出力の設定（4）
    sh = logging.StreamHandler()
    logger.addHandler(sh)


    model_name = 'inception'
    base_model = create_base_network(model_name)
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    model = build_predict(base_model, input_shape=input_shape)
    model.summary()

    dlist_v1 = glob(path.join(sys.argv[1], '*'))
    dlist_v2 = glob(path.join(sys.argv[2], '*'))

    dlist_v1.sort()
    dlist_v2.sort()

    result = []
    # ぜんepochに対応したいときはここをコメントインして下をインデント
    # for epoch in range(int(sys.argv[3])):
        # epoch = str(epoch+1).zfill(2)
        # print('epoch ',epoch)

    logger.log(30, 'epoch {}'.format(epoch))

    model.load_weights('./../model_01/weights.{}.hd5'.format(epoch))

    query = 0
    correct = 0

    for dirs in zip(dlist_v1, dlist_v2):
        flist_v1 = glob(path.join(dirs[0], '*.png'))
        flist_v2 = glob(path.join(dirs[1], '*.png'))
        flist_v1.sort()
        flist_v2.sort()
        # print (len(flist_v1), len(flist_v2))
        logger.log(30, '{} {}'.format(len(flist_v1), len(flist_v2)))
        query += len(flist_v1)

        for i, ref in enumerate(flist_v1):
            ref_img = get_img(ref)
            ref_img = np.expand_dims(ref_img, axis=0)
            r_feat = model.predict(ref_img,batch_size=1)

            min_dist = 1000
            nn = 0
            for j, q in enumerate(flist_v2):
                q_img = get_img(q)
                q_img = np.expand_dims(q_img, axis=0)
                q_feat = model.predict(q_img,batch_size=1)
                dist = np.linalg.norm( r_feat[0] - q_feat[0])
                if dist < min_dist:
                    min_dist = dist
                    nn = j
            # print(i, nn, min_dist)
            logger.log(30, '{} {} {}'.format(i, nn, min_dist))
            if i == nn:
                correct += 1

    if correct == 0:
        logger.log(30, "{} files, {} corrects, accuracy = zero divided".format(query, correct))
        result.append(0)
    else:
        logger.log(30, "{} files, {} corrects, accuracy = {}".format(query, correct, float(correct / query)))
        result.append(float(correct / query))


    logger.log(30, result)

if __name__ == '__main__':
    # train_aug()
    test()

#model.fit_generator(...)
