import filecmp
import os
import sys
from glob import glob

def setup():
    # frames : 見つけたい画像が入ってるはずのフォルダの中のファイル一覧[string]
    print("frame_folder_path")
    dir = input()
    frames = glob(os.path.join(dir, '*'))

def samepic(anno):
    # dir : 見つけたい画像が入ってるはずのフォルダ
    # anno : すでにポジティブでリネームしてしまった画像

    for frame in frames:
        if filecmp.cmp(anno,frame):
            return frame

    print('same picture not found')
    return 'none'
