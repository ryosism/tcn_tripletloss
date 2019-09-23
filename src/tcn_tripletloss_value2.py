from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, merge, Flatten, Embedding, Dense, GlobalAveragePooling2D, Lambda
from keras import backend as K
# import keras
from keras.models import load_model
import random
# import keras.backend.tensorflow_backend as KTF

import numpy as np
from glob import glob
import os
import sys
import time

# import pandas as pd
import logging

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json


# argv[]
# argv[1] : input directory for test_anchor
# argv[2] : input directory for test_positive
# argv[3] : epoch_num(max)
# argv[4] : model(.hd5) directory

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def visualize_vec(vec):
    anchor, positive, negative = vec
    print("anchor = {}".format(anchor))
    # この下にt-sneを書いていきたい

def triplet_loss(vec, alpha = 0.2):
    visualize_vec(vec)

    anchor, positive, negative = vec
    d_p = K.sum(K.square(anchor - positive), axis = -1)
    d_n = K.sum(K.square(anchor - negative), axis = -1)
    loss = K.mean(K.maximum(0.0, d_p -d_n + alpha ))
    return loss

def top5(rank_dist, rank_index, new, index, filename, top5name):
    for i, dist in enumerate(rank_dist):
        if float(new) < dist:
            for frame in rank_index:
                if index in range(frame-30, frame+30):
                    return rank_dist, rank_index, top5name

            rank_dist.insert(i, float(new))
            rank_dist.pop(5)
            rank_index.insert(i, index)
            rank_index.pop(5)
            top5name.insert(i, os.path.abspath(filename))
            top5name.pop(5)

            return rank_dist, rank_index, top5name

    return rank_dist, rank_index, top5name


def top20(rank_dist, rank_index, new, index, filename, top5name):
    for i, dist in enumerate(rank_dist):
        if float(new) < dist:
            for frame in rank_index:
                if index in range(frame-20, frame+20):
                    return rank_dist, rank_index, top5name

            rank_dist.insert(i, float(new))
            rank_dist.pop(20)
            rank_index.insert(i, index)
            rank_index.pop(20)
            top5name.insert(i, os.path.abspath(filename))
            top5name.pop(20)

            return rank_dist, rank_index, top5name

    return rank_dist, rank_index, top5name


def top500(rank_dist, rank_index, new, index):
    for i, score in enumerate(rank_dist):
        if float(new) < score:
            rank_dist.insert(i, float(new))
            rank_dist.pop(500)
            rank_index.insert(i, index)
            rank_index.pop(500)

            return rank_dist, rank_index

    return rank_dist, rank_index

def confidence_graph(confidences, order_num):
    fig = plt.figure(figsize=(200, 10))
    plt.xlabel("frame")
    plt.ylabel("confidence")
    plt.rcParams["font.size"] = 16
    # plt.legend(["confidence"], loc= loc="upper right")

    frames = range(len(confidences))
    num = 100 #移動平均の個数
    b = np.ones(int(num))/num
    ido = np.convolve(confidences, b, mode='same')#移動平均

    plt.xlim(0, len(confidences))
    plt.ylim(0.5, 1.0)

    plt.plot(frames, confidences, "r",linewidth=1, linestyle="-", label="conf")
    plt.plot(frames, ido, "b",linewidth=10, linestyle="-", label="moving average (n = {})".format(int(num)))
    plt.legend(fontsize=18)

    plt.savefig("confidence_order{}.pdf".format(order_num),bbox_inches="tight")


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
    now = time.ctime()
    parsed = time.strptime(now)
    fh = logging.FileHandler('output_value2_{}.log'.format(time.strftime("%Y%m%d_%H:%M:%S", parsed)))
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

    # glob対象が複数の場合はこっち
    # dlist_v1 = glob(os.path.join(sys.argv[1], '*'))
    # dlist_v2 = glob(os.path.join(sys.argv[2], '*'))

    # glob対象が単一ディレクトリの場合はこっち
    dlist_v1 = []
    dlist_v2 = []
    dlist_v1.append(sys.argv[1])
    dlist_v2.append(sys.argv[2])

    dlist_v1.sort()
    dlist_v2.sort()

    key_indexs = []
    top5Corrects = []
    allepoch_top5filenames = []

    # 全部のepochいじるときはfor文を有効に
    for epoch in range(int(sys.argv[3])):
        epoch = str(epoch+1).zfill(2)
        logger.log(30, 'epoch {}'.format(epoch))

        model.load_weights('{}/weights.{}.hd5'.format(sys.argv[4], epoch))

        query = 0
        correct = 0
        all_distances = []
        all_confidences = []
        epoch_top5Indexes = []
        epoch_top5filenames = []

        for dirs in zip(dlist_v1, dlist_v2):
            flist_v1 = glob(os.path.join(dirs[0], '*.png'))
            flist_v2 = glob(os.path.join(dirs[1], '*.png'))
            flist_v1.sort()
            flist_v2.sort()
            logger.log(30, '{} {}'.format(len(flist_v1), len(flist_v2)))
            query += len(flist_v1)

            q_feats = []
            for index, img in enumerate(flist_v2):
                if index%500 == 0:
                    # print("flist_v2", j)
                    logger.log(30, "flist_v2 {}".format(index))
                q_img = get_img(img)
                q_img = np.expand_dims(q_img, axis=0)
                q_feat = model.predict(q_img,batch_size=1)
                q_feats.append(q_feat)

            min_index = 0
            file = open("query.json", "w")
            json.dump(flist_v1, file)
            # -----------------------------------
            for i, ref in enumerate(flist_v1):
                # 手順画像ごとに実行

                # top5
                rank_dist = [1000,1000,1000,1000,1000]
                rank_index = [0,0,0,0,0]
                top5_filename = ["","","","",""]

                # top20
                # rank_dist = []
                # rank_index = []
                # top5_filename = []
                # for i in range(20):
                #     rank_dist.append(100)
                #     rank_index.append(0)
                #     top5_filename.append("")

                # -----------------------------

                ref_img = get_img(ref)
                ref_img = np.expand_dims(ref_img, axis=0)
                r_feat = model.predict(ref_img,batch_size=1)

                min_dist = 1000
                nn = 0
                confidences = []
                distances = []

                for j in range(min_index, len(q_feats)):
                    dist = np.linalg.norm( r_feat[0] - q_feats[j][0])

                    # 手順画像に対してフレーム１枚ごとに実行
                    # フレーム数分だけforが回る

                    # top1
                    if dist < min_dist:
                        min_dist = dist
                        nn = j

                    distances.append(dist)
                    confidences.append(1.0 - dist)

                    # top5
                    rank_dist, rank_index, top5_filename = top5(rank_dist, rank_index, dist, j, flist_v2[j], top5_filename)

                # confidence_graph(confidences, i)
                all_distances.append(distances)
                all_confidences.append(confidences)

                na = np.array(all_confidences, dtype=np.float)

                # top1
                logger.log(30, '{} {} {}'.format(i+1, nn, min_dist))
                file_index = (os.path.basename(flist_v2[nn])).split('.')[0]
                key_indexs.append(int(file_index))
                logger.log(30, 'query = {}'.format(ref))
                logger.log(30, 'top1 = {}'.format(flist_v2[nn]))

                # top5
                logger.log(30, 'rank_dist = {} \nrank_index = {}'.format(rank_dist, rank_index))
                epoch_top5Indexes.append(rank_index)
                epoch_top5filenames.append(top5_filename)
                for i, name in enumerate(top5_filename):
                    logger.log(30, "top{} {}".format(i+1, name))

                # このインデントは１つの手順画像が終わるごと
            # このインデントは全部の手順画像が終わったとき
        # このインデントはepochが終わった時
        allepoch_top5filenames.append(epoch_top5filenames)
        # logger.log(30, allepoch_top5filenames)
    # このインデントは全てのepochが終わった時

            # path = nx.dijkstra_path(make_tensor(all_distances, len(flist_v1), 0.6), 0, len(flist_v1)*len(flist_v2))

            # order_path = []
            # now = -1
            # for i, pa in enumerate(path):
            #     pa = pa % len(flist_v1)
            #     if now != pa:
            #         now = pa
            #         print(i)
            #         order_path.append(pa)
            #
            # print("order_path = ",order_path)

        logger.log(30,"key_indexs = {}".format(key_indexs))

        # import value2_top5_186 as value186
        # import value2_top5_112 as value112
        # import value2_top5_186_017 as value186_017
        # import value2_top5_112_017 as value112_017
        #
        # top5Correct = value186.calcAccuracy(epoch_top5Indexes, logger)
        # top5Correct = value112.calcAccuracy(epoch_top5Indexes, logger)
        # top5Correct = value186_017.calcAccuracy(epoch_top5Indexes, logger)
        # top5Correct = value112_017.calcAccuracy(epoch_top5Indexes, logger)

        # logger.log(30, "top5_correct = {}".format(top5Correct))
        # top5Corrects.append(top5Correct)
        # logger.log(30, top5Corrects)

    file = open("candidate.json", "w")
    json.dump(allepoch_top5filenames, file)



if __name__ == '__main__':
    # train_aug()
    test()

#model.fit_generator(...)
