import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json


def parseJSON(predicted, answer):
    with open(predicted, "r") as f:
        predicted = json.load(f)

    with open(answer, "r") as f:
        answer = json.load(f)

    return predicted, answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted', '-p', default='', help='Path_to_predicted_JSON')
    parser.add_argument('--answer', '-a', default='', help='Path_to_answer_JSON')
    parser.add_argument('--outputPath', '-o', default='./graph.pdf', help='Path_to_graphImage')
    return parser.parse_args()


def calc(pre, gtmin, gtmax):
    '''
        スコア定義基準
        1) 推測されたフレーム(pre)が正解の範囲の下限(gtmin)と上限(gtmax)の中にあるなら，score = 1.0
        2) ないなら，正解範囲からどれだけ外れているか(diff)を求め，diffが大きくなるにつれscoreを1.0から減らす
        3) diffの許容値は1分40秒(100sec, 3000frames)とする，これはシステムが+10秒，-10秒のボタンを押す回数が10回を超えるとUXが低下するという仮定の元設定した
    '''
    pre, gtmin, gtmax = int(pre), int(gtmin), int(gtmax)
    if gtmin <= pre <= gtmax:
        score = 1.0
    elif pre < gtmin:
        diff = gtmin - pre
        score = max(0, 1.0 - (diff/3000))
    else: # gtmax < pre
        diff = pre - gtmax
        score = max(0, 1.0 - (diff/3000))

    return score


def eval(args):
    predict, answer = parseJSON(predicted = args.predicted, answer = args.answer)

    maxEpochScores = []
    for epochNum, epoch in enumerate(predict):
        epochScores = []
        for idx, order in enumerate(epoch):
            scores = []
            for candidate in order:
                frameNum = candidate.split('/')[-1].split('.')[0]
                gtmin = answer[idx]['gtmin']
                gtmax = answer[idx]['gtmax']
                scores.append(calc(pre=frameNum, gtmin=gtmin, gtmax=gtmax))

            # １手順の複数の候補の中で一番高いスコアを取得
            maxScore = max(scores)
            epochScores.append(maxScore)

        # 各手順のスコアの平均
        maxEpochScore = round(np.mean(epochScores), 4)

        print("epoch {}: score = {}".format(epochNum, maxEpochScore))
        maxEpochScores.append(maxEpochScore)

    plot(maxEpochScores=maxEpochScores, outputPath=args.outputPath)
    print(maxEpochScores)


def plot(maxEpochScores, outputPath):
    plt.figure(figsize=(16, 12))
    x = range(len(maxEpochScores))
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.plot(x, maxEpochScores ,"r",linewidth=1.5, linestyle="-")
    plt.savefig(outputPath, bbox_inches="tight")


if __name__ == '__main__':
    args = parse_args()
    eval(args)
