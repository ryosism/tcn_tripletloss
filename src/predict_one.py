from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, merge, Flatten, Embedding, Dense, GlobalAveragePooling2D, Lambda
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.applications.inception_v3 import InceptionV3

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
sys.argv[2] : 特徴を取りたい画像のパス

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

    # モデルに画像を通す
    feat = model.predict(np.expand_dims(get_img(sys.argv[2]), axis=0), batch_size=1)
    print(feat)
    print(feat.shape)

    feat = feat.astype(np.float64)
    Y = tsne(feat, perplexity=1000.0)
    plt.scatter(Y[:, 0], Y[:, 1], c=feat.target)
    plt.savefig("pic_feature.pdf", bbox_inches="tight")
