# -*- coding: utf-8 -*-
#
# K-meansによる手書き文字分類
#
# 2016/08/08 ver1.1 07-mix_em.pyに06-k_means.pyのコード取り込み by Itsukara
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pandas import Series, DataFrame
from numpy.random import randint, rand

#------------#
# Parameters #
#------------#
K  = 10   # 分類する文字数
N  = 20   # 反復回数
C  = 20   # 図の表示カラム数
NS =600   # 初期データを生成する際に使うサンプル数(0:サンプル未使用)

# 分類結果の表示
def show_figure(mu, cls):
    fig = plt.figure(figsize=(16,K))
    for c in range(K):
        subplot = fig.add_subplot(K,C,c*C+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('Master')
        subplot.imshow(mu[c].reshape(28,28), cmap=plt.cm.gray_r)
        i = 1
        for j in range(len(cls)):
            if cls[j] == c:
                subplot = fig.add_subplot(K,C,c*C+i+1)
                subplot.set_xticks([])
                subplot.set_yticks([])
                subplot.imshow(df.ix[j].reshape(28,28), cmap=plt.cm.gray_r)
                i += 1
                if i > C - 1:
                    break
    fig.show()
    fig.savefig("kmeans-figure_2-K{}-NS{}-{}.png".format(K, NS, time.asctime()[11:19]))


# Main
if __name__ == '__main__':
    # トレーニングセットの読み込み
    df = pd.read_csv('sample-images.txt', sep=",", header=None)

    labels = pd.read_csv('sample-labels.txt', header=None)
    labels = np.array(labels).flatten()
    unique_labels = np.unique(labels)
    unique_labels_list = list(unique_labels)
    start_time = time.time()
    print("07-k_means.py: K={}, N={}, C={}, NS={}".format(K, N, C, NS))

    # 初期パラメータの設定
    center = (rand(28*28*K)*0.5+0.25).reshape(K, 28*28)
    if NS > 0:
        for clsi, label in enumerate(unique_labels):
            samples = df[labels == label]
            if NS < len(samples):
                samples = samples.sample(NS)
            center[clsi] = np.array(samples).mean(axis=0)

    fig = plt.figure(figsize=(16,K))
    for k in range(K):
        subplot = fig.add_subplot(K, N+1, k*(N+1)+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(center[k].reshape(28,28), cmap=plt.cm.gray_r)
    fig.show()

    cls = [0] * len(df)
    distortion = 0.0

    # N回のIterationを実施
    for iter_num in range(N):
        print("iter_num %d" % iter_num)

        center_new = []
        for k in range(K):
            center_new.append(np.zeros(28*28))
        center_new = np.array(center_new)
        num_points = np.zeros(K, dtype=int)
        distortion_new = 0.0

        # E Phase: 各データが属するグループ（代表data）を計算
        for index, line in df.iterrows():
            min_dist = 28.0 * 28.0
            for i in range(K):
                diff = line - center[i]
                d = sum(diff * diff)
                if d < min_dist:
                    min_dist = d
                    cls[index] = i
            center_new[cls[index]] += line
            num_points[cls[index]] += 1
            distortion_new += min_dist

        # M Phase: 新しい代表imageを計算
        for k in range(K):
            if num_points[k] != 0:
                center_new[k] = center_new[k] / num_points[k]
            else:
                print("[WARNING] num_point[{}] == 0".format(k))
            subplot = fig.add_subplot(K, N+1, k*(N+1)+(iter_num+1)+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.imshow(center_new[k].reshape(28,28), cmap=plt.cm.gray_r)
        center = center_new
        print("Distortion: J=%d" % distortion_new)

        fig.show()

    fig.savefig("kmeans-figure_1-K{}-NS{}-{}.png".format(K, NS, time.asctime()[11:19]))

    print("Elapsed = {:.2f} sec.".format(time.time() - start_time))

    # 分類結果の表示
    show_figure(center, cls)

    def print_histogram(clsi):
        num_labels = {}
        for label in unique_labels:
            num_labels[label] = 0
        for index in range(len(df)):
            if cls[index] == clsi:
                num_labels[labels[index]] += 1
        print("[Cluster{:02d}] ".format(clsi), end='') 
        for label, num in num_labels.items():
            print("{:4d}  ".format(num), end='') 
        print()

    if NS > 0:
        num_samples = np.zeros(K)
        num_correct = np.zeros(K)
        for index in range(len(df)):
            clsi = cls[index]
            num_samples[clsi] += 1
            if clsi < len(unique_labels):
                if labels[index] == unique_labels[clsi]:
                    num_correct[clsi] += 1

        num_correct_sum = 0
        for clsi in range(len(unique_labels)):
            print("[Label{}] #members={:3.0f}, #correct-members={:3.0f}, accuracy={:.1f}%".format(
                    unique_labels[clsi], num_samples[clsi], num_correct[clsi],
                    100.0 * num_correct[clsi] / num_samples[clsi]))
            num_correct_sum += num_correct[clsi]

        print("[TOTAL ] #members={:3.0f}, #correct-members={:3.0f}, accuracy={:.1f}%".format(
                len(df), num_correct_sum,
                100.0 * num_correct_sum / len(df)))

    print("***** Histogram of labels *****")
    print("            ", end='')
    for label in unique_labels:
        print("{:4d}  ".format(label), end='') 
    print()
    print("-"*80)
    for clsi in range(K):
        print_histogram(clsi)

