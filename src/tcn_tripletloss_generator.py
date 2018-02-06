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
import os
import sys

import json
import time

import cv2

# argv[]
# argv[1] : json directoried
# argv[2] : epoch_num

# if you wanna train, please add "train_aug()" after "if __name__ == '__main__':" .

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def triplet_loss(vec, alpha = 0.2):
    beta = 1.0
    anchor, positive, negative = vec
    d_p = K.sum(K.square(anchor - positive), axis = -1)
    d_n = K.sum(K.square(anchor - negative), axis = -1)
    loss = K.mean(K.maximum(0.0, d_p -d_n + alpha )) # + beta * (K.mean(K.minimum(0.0, d_p - 0.1)))
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
    if random.randint(0, 1):
        x = cv2.flip(x, 1)
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


def random_batch_generator(all_frame, train_list, num_batches = 32):
    while True:
        anc = []
        pos = []
        neg = []
        np.random.shuffle(train_list)
        for i in range(len(train_list)):
            frame = random.randint(0, len(train_list[i])-1)
            anchor = train_list[i][frame][0]
            positive = train_list[i][frame][1]
            # 上のpositiveはファイルパスのはず
            # 何番の料理を見てるか番号取得
            path = positive.split('/')
            # path[-2]が番号かな、[-1]がframe、[0]がファイル名
            # 配列番号を取得
            folder_num = int(path[-3])
            frames = all_frame[folder_num-1]
            # これでframesはpositiveと同じ動画のフレーム画像リスト

            # 拡張子とファイル名を分けて、positiveが見てるファイルのフレーム番号を取得
            name, ext = os.path.splitext(path[-1])
            po_frameNum = int(name)

            while True:
                # nはネガティブ候補のフレーム番号
                n = random.randint(0, len(frames)-1)
                # print("n = {}, po_frameNum = {}".format(n, po_frameNum))
                if abs(n - po_frameNum) > 301:
                    break
            negative = frames[n]
            pos.append(get_img(positive))
            anc.append(get_img(anchor))
            neg.append(get_img(negative))

            if len(anc) == num_batches:
                yield({'input_positive':np.asarray(pos, dtype=np.float32), 'input_negative':np.asarray(neg, dtype=np.float32), 'input_anchor':np.asarray(anc, dtype=np.float32)},{'triplet_loss':np.zeros((num_batches, 1), dtype = np.float32)})
                pos = []
                anc = []
                neg = []

def train_aug():

    # 全フレーム取得、リストで
    all_frame = []
    # 柔軟性ないコードだけどまぁいいや
    for i in range(10):
        path = '../../dataset/RakutenDS/triplet/{}/frame/'.format(str(i+1).zfill(3))
        frames = glob(os.path.join(path, '*'))
        all_frame.append(frames)


    model_name = 'inception'
    base_model = create_base_network(model_name)
    base_model.summary()

    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    model = build(base_model, input_shape=input_shape)
    model.summary()
    model.compile(optimizer='rmsprop', loss=identity_loss)

    model_arch = base_model.to_json()
    fout = open("base_model.json", 'w').write(model_arch)

    model_arch = model.to_json()
    fout = open("model.json", 'w').write(model_arch)

    jlist_v1 = glob(os.path.join(sys.argv[1], '*'))
    jlist_v1.sort()

    train_list =[]
    total_recipes = 0
    total_frames = 0

    # jsonファイルごとに
    for jfile in jlist_v1:
        # 読みこみ
        f = open(jfile, "r")
        JSON = json.load(f)
        # レシピIDごとに
        for idnum in range(len(JSON['data'])):
            name_list =[]
            # 1ペアごとに
            for pic in (JSON['data'][idnum][1]['pic']):
                # オーギュメンテーション
                # -30フレームから30フレーム後まで
                for t in range(-300,300,60):
                    # dictにファイルパス、fileにファイル名が入る
                    po_path, file = os.path.split(pic['positive'])
                    aug_po = po_path + '/' + str(pic['positive_index']+t).zfill(5) + '.png'
                    print(pic['anchor'], aug_po)
                    if os.path.exists(aug_po):
                        name_list.append((pic['anchor'], aug_po))

                        train_list.append(name_list)
                        total_frames += len(name_list)

            total_recipes += 1
    print("len(train_list), total_recipes = ", len(train_list), total_recipes)
    batchsize = 50
    train_epoch = int(sys.argv[2])
    out_model_path = './../model_14/model/weights.{epoch:02d}.hd5'
    checkpoint = keras.callbacks.ModelCheckpoint(out_model_path, verbose = 1)
    tensorboard = keras.callbacks.TensorBoard(log_dir="./log/", write_graph=True)

    now = time.ctime()
    parsed = time.strptime(now)
    csvlogger = keras.callbacks.CSVLogger('{}.csv'.format(time.strftime("%Y%m%d_%H:%M:%S", parsed)), separator=',', append=False)
    logs = model.fit_generator(random_batch_generator(all_frame, train_list, batchsize), steps_per_epoch = total_frames/batchsize, epochs = train_epoch, callbacks=[checkpoint, tensorboard, csvlogger])

#
# def test():
#     query = 0
#     correct = 0
#
#     model_name = 'inception'
#     base_model = create_base_network(model_name)
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, 224, 224)
#     else:
#         input_shape = (224, 224, 3)
#     model = build_predict(base_model, input_shape=input_shape)
#     model.summary()
#     model.load_weights('./../model/weights.198.hd5')
#
#     dlist_v1 = glob(os.path.join(sys.argv[4], '*'))
#     dlist_v2 = glob(os.path.join(sys.argv[5], '*'))
#
#     dlist_v1.sort()
#     dlist_v2.sort()
#
#     for dirs in zip(dlist_v1, dlist_v2):
#         flist_v1 = glob(os.path.join(dirs[0], '*.png'))
#         flist_v2 = glob(os.path.join(dirs[1], '*.png'))
#         flist_v1.sort()
#         flist_v2.sort()
#         print (len(flist_v1), len(flist_v2))
#         query += len(flist_v1)
#         for i, ref in enumerate(flist_v1):
#             ref_img = get_img(ref)
#             ref_img = np.expand_dims(ref_img, axis=0)
#             r_feat = model.predict(ref_img,batch_size=1)
#             #print("r_feat = ",np.asarray(r_feat).shape)
#             #print (r_feat)
#             min_dist = 1000
#             nn = 0
#             for j, q in enumerate(flist_v2):
#                 q_img = get_img(q)
#                 q_img = np.expand_dims(q_img, axis=0)
#                 q_feat = model.predict(q_img,batch_size=1)
#                 dist = np.linalg.norm( r_feat[0] - q_feat[0])
#                 if dist < min_dist:
#                     min_dist = dist
#                     nn = j
#             print(i, nn, min_dist)
#             if i == nn:
#                 correct += 1
#
#     print("{} files, {} corrects, test_loss = {}".format(query, correct, 1-float(correct / query)))

if __name__ == '__main__':
    train_aug()
    # test()

#model.fit_generator(...)
