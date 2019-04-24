from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, merge, Flatten, Embedding, Dense, GlobalAveragePooling2D, Lambda, Dropout
from keras import backend as K
import keras
import random
import keras.backend.tensorflow_backend as KTF

import numpy as np
from glob import glob
import os
import sys

import json
import time

import cv2

# debuggers
import pdb
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

# argv[]
# argv[1] : json directoried
# argv[2] : epoch_num


def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)


def whatIs_y_true(y_true, y_pred):
    return y_true


def whatIs_y_pred(y_true, y_pred):
    return y_pred


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def triplet_loss(vec, alpha=1.0):
    anchor, positive, negative = vec
    d_p = K.sum(K.square(anchor - positive), axis=-1)
    d_n = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.mean(K.maximum(0.0, d_p - d_n + alpha))
    return loss


def create_base_network(model_name):
    model_name = 'inception'
    if model_name == 'vgg16':
        vgg_model = VGG16(include_top=False, weights='imagenet')
        model = Model(input=vgg_model.input, output=vgg_model.output)
    elif model_name == 'inception':
        inception_model = InceptionV3(weights='imagenet', include_top=False)
        x = inception_model.get_layer('mixed5').output
        # x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)

        x = Dense(128)(x)
        model = Model(input=inception_model.input, output=x)
    return model


def triplet_output_shape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def build(base_model, input_shape=(224, 224, 3)):
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


def get_img(name):
    img = image.load_img(name, target_size=(224, 224))
    x = image.img_to_array(img)
    if random.randint(0, 1):
        x = cv2.flip(x, 1)
    return x


def random_batch_generator(all_frame, train_list, num_batches=64):
    print('random_batch_generator called')
    flag = True
    while True:
        anc = []
        pos = []
        neg = []
        np.random.shuffle(train_list)
        for i in range(len(train_list)): # DSの回数だけ繰り返す
            frame = random.randint(0, len(train_list[i]) - 1)
            anchor = train_list[i][frame][0]
            positive = train_list[i][frame][1]
            # 上のpositiveはファイルパスのはず
            # 何番の料理を見てるか番号取得
            path = positive.split('/')
            # path[-2]が番号かな、[-1]がframe、[0]がファイル名
            # 配列番号を取得
            folder_num = int(path[-3])
            frames = all_frame[folder_num - 1]
            # これでframesはpositiveと同じ動画のフレーム画像リスト

            # 拡張子とファイル名を分けて、positiveが見てるファイルのフレーム番号を取得
            name, ext = os.path.splitext(path[-1])
            po_frameNum = int(name)

            # if np.random.randint(0, 1):
            #     while True:
            #         # nはネガティブ候補のフレーム番号
            #         n = random.randint(0, len(frames)-1)
            #         # print("n = {}, po_frameNum = {}".format(n, po_frameNum))
            #         if abs(n - po_frameNum) > 300:
            #             break
            #     negative = frames[n]
            # else:
            #     # 他の料理の動画の全フレームもnegative対象にする
            #     # まずハンバーグだけ別処理にする(ハンバーグは別のフォルダにもフレームが存在するから)
            #     if folder_num == 5 or folder_num == 11 or folder_num == 12 or folder_num == 13:
            #         array = [1,2,3,4,6,7,8,9,10] #ハンバーグじゃないやつ
            #         folder_num = np.random.choice(array)
            #         frames = all_frame[folder_num-1]

            while True:
                # nはネガティブ候補のフレーム番号
                n = random.randint(0, len(frames) - 1)
                # print("n = {}, po_frameNum = {}".format(n, po_frameNum))
                if 1000 > abs(n - po_frameNum) > 300:
                    break
            negative = frames[n]

            pos.append(get_img(positive))
            anc.append(get_img(anchor))
            neg.append(get_img(negative))

            if len(anc) == num_batches:
                if flag:
                    print("anchor[0] = {}".format(anchor))
                    print("positive[0] = {}".format(positive))
                    print("negative[0] = {}".format(negative))
                    flag = False

                yield(
                    {
                        'input_positive': np.asarray(pos, dtype=np.float32),
                        'input_negative': np.asarray(neg, dtype=np.float32),
                        'input_anchor': np.asarray(anc, dtype=np.float32)
                    },
                    {
                        'triplet_loss': np.zeros((num_batches, 3), dtype=np.float32)
                    }
                )
                pos = []
                anc = []
                neg = []


def train_aug():

    # 全フレーム取得、リストで
    all_frame = []
    # 柔軟性ないコードだけどまぁいいや
    for i in range(15):
        path = '../../dataset/RakutenDS/triplet/{}/frame/'.format(str(i + 1).zfill(3))
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
    model.compile(optimizer=keras.optimizers.Adamax(), loss=identity_loss, metrics=[whatIs_y_pred, whatIs_y_true])

    model_arch = base_model.to_json()
    fout = open("base_model.json", 'w').write(model_arch)

    model_arch = model.to_json()
    fout = open("model.json", 'w').write(model_arch)

    jlist_v1 = glob(os.path.join(sys.argv[1], '*'))
    jlist_v1.sort()

    train_list = []
    total_recipes = 0
    total_frames = 0

    # jsonファイルごとに
    for jfile in jlist_v1:
        # 読みこみ
        f = open(jfile, "r")
        JSON = json.load(f)
        # レシピIDごとに
        for idnum in range(len(JSON['data'])):
            name_list = []
            # 1ペアごとに
            for pic in (JSON['data'][idnum][1]['pic']):
                # オーギュメンテーション
                # -30フレームから30フレーム後まで
                for t in range(-50, 50, 10):
                    # dictにファイルパス、fileにファイル名が入る
                    po_path, file = os.path.split(pic['positive'])
                    aug_po = po_path + '/' + str(pic['positive_index'] + t).zfill(5) + '.png'
                    print("anchor : {}, positive : {}".format(pic['anchor'], aug_po))
                    if os.path.exists(aug_po):
                        name_list.append((pic['anchor'], aug_po))

                        train_list.append(name_list)
                    else:
                        print('not found : {}'.format(aug_po))

            total_recipes += 1
    print("len(train_list), total_recipes = ", len(train_list), total_recipes)
    batchsize = 50

    tensorboard = keras.callbacks.TensorBoard(log_dir="./tensorboard/logs/", write_graph=True, write_images=1)

    now = time.ctime()
    parsed = time.strptime(now)

    csvlogger = keras.callbacks.CSVLogger(
        './csv/{}.csv'.format(time.strftime("%Y%m%d_%H:%M:%S", parsed)), separator=',', append=False)
    history = keras.callbacks.History()

    batch_print_callback = keras.callbacks.LambdaCallback(
        on_batch_end=lambda batch, logs: print("\nbatch = {},  logs = {}".format(batch, logs)))
    pdb_callback = keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: pdb.set_trace())

    train_epoch = int(sys.argv[2])
    for i in range(train_epoch):

        now = time.ctime()

        out_model_path = '/root/ex29/model/weights.{}.hd5'.format(str(i).zfill(2))
        checkpoint = keras.callbacks.ModelCheckpoint(out_model_path, verbose=1)
        train_sample = random.sample(train_list, k=int(len(train_list)*0.8))
        logs = model.fit_generator(
            random_batch_generator(all_frame, train_list, batchsize),
            initial_epoch=i,
            steps_per_epoch=len(train_list) / batchsize,
            epochs=i + 1,
            callbacks=[checkpoint, tensorboard, csvlogger, history]
        )

        parsed = time.strptime(now)
        print('{}'.format(time.strftime("%Y%m%d_%H:%M:%S", parsed)))

        print("logs = ", logs)


if __name__ == '__main__':
    train_aug()
