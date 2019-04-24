from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, merge, Flatten, Embedding, Dense, GlobalAveragePooling2D, Lambda
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.applications.inception_v3 import InceptionV3

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

import cv2
import sys
import os
import numpy as np
from glob import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from bhtsne import tsne
from sklearn.datasets import load_iris


'''
sys.argv[1] : 学習済みモデルのパス
sys.argv[2] : 特徴を取りたい画像のディレクトリパス
sys.argv[3] : test_anchorの画像、他とは区別して表示させたい画像
sys.argv[4] : 散布図の保存ファイル名

'''


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def visualize_vec(vec):
    anchor, positive, negative = vec
    print("anchor = {}".format(anchor))
    # この下にt-sneを書いていきたい


def triplet_loss(vec, alpha=0.2):
    visualize_vec(vec)

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
        x = GlobalAveragePooling2D()(x)

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


def build_predict(base_model, input_shape=(224, 224, 3)):
    input_img = Input(shape=input_shape)
    out = base_model(input_img)
    return Model(input=input_img, output=out)


def get_img(name):
    img = image.load_img(name, target_size=(224, 224))
    x = image.img_to_array(img)
    return x


def scatter_image(filename):
    """
    Args:
        feature_x: x座標
        feature_y: y座標
        image_paths:
    """

    feature_x = Y[:, 0]
    feature_y = Y[:, 1]
    image_paths = imgList

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xlim = [feature_x.min()-10, feature_x.max()+10]
    ylim = [feature_y.min()-10, feature_y.max()+10]

    for (x, y, path) in zip(feature_x, feature_y, image_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bb = Bbox.from_bounds(x, y, 3, 3)
        bb2 = TransformedBbox(bb, ax.transData)
        bbox_image = BboxImage(bb2, norm=None, origin=None, clip_on=False)
        bbox_image.set_data(img)
        ax.add_artist(bbox_image)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    plt.savefig(filename, dpi=1200)


# 間引き
def thinouted(lst, newsize):
    """間引いたリストを返す"""
    result = []
    cnt = 0
    for x in lst:
        cnt -= newsize
        if cnt < 0:
            cnt += len(lst)
            result.append(x)
    return result
# lst = thinouted(range(100), 4)



if __name__ == '__main__':

    # セットアップ
    model_name = 'inception'
    base_model = create_base_network(model_name)
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    model = build_predict(base_model, input_shape=input_shape)
    model.summary()

    # 学習で出力されたjsonファイルと重みをロード
    model.load_weights(sys.argv[1])

    # 画像リストの取得
    imgList = glob(os.path.join(sys.argv[2], '*'))

    # 間引きする
    imgList = thinouted(imgList, 200)

    # もらったtest_anchorの画像をリストの末尾に追加
    ancList = glob(os.path.join(sys.argv[3], '*'))
    imgList += ancList

    # モデルに画像を通す
    featList = []
    for index, imgPath in enumerate(imgList):
        feat = model.predict(np.expand_dims(get_img(imgPath), axis=0), batch_size=50)
        feat = feat.astype(np.float64)
        featList.append(feat)
        if index % 500 == 0:
            print('{} images loaded'.format(index))

    featList = np.array(featList)
    featList = featList[0:featList.shape[0], 0, 0:featList.shape[2]]

    # label = []
    # classlabel = 0
    # for i in range(index + 1):
    #     label.append(classlabel)
    #     if i % 250 == 249:
    #         classlabel += 1

    Y = tsne(featList, perplexity=5)

    # 動画フレームplot
    # plt.scatter(Y[0:2500, 0], Y[0:2500, 1], c=range(len(imgList) - len(ancList)), s=5)
    # plt.colorbar()

    # 手順画像plot
    # plt.scatter(Y[2500:, 0], Y[2500:, 1], c=range(len(ancList)), s=80, marker='x', edgecolors='red', linewidths='10')
    # plt.colorbar()

    # プロットした点の右上に手順番号を付加
    # for index in range(len(ancList)):
    #     plt.annotate(str(index), (Y[2500 + index, 0], Y[2500 + index, 1]))

    # plt.savefig(sys.argv[4], bbox_inches="tight")

    scatter_image(filename = "{}".format(sys.argv[4]))
