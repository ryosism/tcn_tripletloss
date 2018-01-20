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
import sys

import json
import time

# argv[]
# argv[1] : json directoried
# argv[2] : epoch_num

# if you wanna train, please add "train_aug()" after "if __name__ == '__main__':" .

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
        anc = []
        pos = []
        neg = []
        np.random.shuffle(train_list)
        for i in range(len(train_list)):
            frame = random.randint(0, len(train_list[i])-1)
            anchor = train_list[i][frame][0]
            positive = train_list[i][frame][1]
            while True:
               n = random.randint(0, len(train_list[i])-1)
               if abs(n - frame) < 30:
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


def train():
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
    model_arch = json_string = model.to_json()
    fout = open("model.json", 'w')
    fout.write(model_arch)
    fout.close()

# ネガ、ポジそれぞれのフォルダをまとめたフォルダのパス
# v1にアンカー、v2にポジティブを置けばいいのかな
    dlist_v1 = glob(sys.argv[1])
    dlist_v2 = glob(sys.argv[2])

# ディレクトリ名でソート
    dlist_v1.sort()
    dlist_v2.sort()

    train_list =[]
    for dirs in zip(dlist_v1, dlist_v2):
        # ディレクトリの中身の画像ファイルのパスを取得
        flist_v1 = glob(path.join(dirs[0], '*/*.png'))
        flist_v2 = glob(path.join(dirs[1], '*/*.png'))
        # ファイル名ソート
        flist_v1.sort()
        flist_v2.sort()

        # アンカーに対してのネガティブペアを作るため、見当違いのインデックス番号のファイルパスをネガティブ画像としてリスト作成
        for p in range( min(len(flist_v1), len(flist_v2))):
            anchor = flist_v1[p]
            positive = flist_v2[p]
            while True:
               n = np.random.randint(0, len(flist_v1))
               if abs(n - p) > 30:
                   break
            negative = flist_v1[n]
            pairs = {'positive':positive, 'negative':negative, 'anchor':anchor}
            train_list.append(pairs)


    batchsize = 32
    train_epoch = 100
    out_model_path = './../model/weights.{epoch:02d}.hd5'
    np.random.shuffle(train_list)
    checkpoint = keras.callbacks.ModelCheckpoint(out_model_path, verbose = 1)
    logs = model.fit_generator(random_batch_generator(train_list, batchsize), steps_per_epoch = len(train_list)/batchsize, epochs = train_epoch, callbacks=[checkpoint])

def train_aug():
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

    jlist_v1 = glob(path.join(sys.argv[1], '*'))
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
                for t in range(-30,30,1):
                    # dictにファイルパス、fileにファイル名が入る
                    po_path, file = path.split(pic['positive'])
                    po_index = pic['positive_index']
                    aug_po = po_path + '/' + str(pic['positive_index']+t).zfill(5) + '.png'
                    print(pic['anchor'], aug_po)
                    if path.exists(aug_po):
                        name_list.append((pic['anchor'], aug_po))

                        train_list.append(name_list)
                        total_frames += len(name_list)

            total_recipes += 1
    print("len(train_list), total_recipes = ", len(train_list), total_recipes)
    batchsize = 64
    train_epoch = int(sys.argv[2])
    out_model_path = './../model/weights.{epoch:02d}.hd5'
    checkpoint = keras.callbacks.ModelCheckpoint(out_model_path, verbose = 1)
    tensorboard = keras.callbacks.TensorBoard(log_dir="./log/", write_graph=True)

    now = time.ctime()
    parsed = time.strptime(now)
    csvlogger = keras.callbacks.CSVLogger('{}.csv'.format(time.strftime("%Y%m%d_%H:%M:%S", parsed)), separator=',', append=False)
    logs = model.fit_generator(random_batch_generator(train_list, batchsize), steps_per_epoch = total_frames/batchsize, epochs = train_epoch, callbacks=[checkpoint, tensorboard, csvlogger])


if __name__ == '__main__':
    train_aug()

#model.fit_generator(...)
