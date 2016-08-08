# -*- coding: utf-8 -*-
#
# 混合ベルヌーイ分布による手書き文字分類
#
# 2015/04/24 ver1.0
# 2016/08/08 ver1.1 サンプルによる初期データ生成、各種表示を追加 by Itsukara
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
    fig.savefig("mixem-figure_2-K{}-NS{}-{}.png".format(K, NS, time.asctime()[11:19]))

# ベルヌーイ分布
def bern(x, mu):
    if False:
#   Slow code (260 sec when K=14)
        r = 1.0
        for x_i, mu_i in zip(x, mu):
            if x_i == 1:
                r *= mu_i 
            else:
                r *= (1.0 - mu_i)
    else:
#   Fast code (70 sec when K=14)
        x = np.array(x)
        r1 = mu[x == 1].prod()
        r2 = (1.0 - mu[x != 1]).prod()
        r = r1 * r2
    return r


# Main
if __name__ == '__main__':
    # トレーニングセットの読み込み
    df = pd.read_csv('sample-images.txt', sep=",", header=None)
    data_num = len(df)

    labels = pd.read_csv('sample-labels.txt', header=None)
    labels = np.array(labels).flatten()
    unique_labels = np.unique(labels)
    unique_labels_list = list(unique_labels)
    start_time = time.time()
    print("07-k_means.py: K={}, N={}, C={}, NS={}".format(K, N, C, NS))

    # 初期パラメータの設定
    mix = [1.0/K] * K
    mu = (rand(28*28*K)*0.5+0.25).reshape(K, 28*28)
    if NS > 0:
        for clsi, label in enumerate(unique_labels):
            samples = df[labels == label]
            if NS < len(samples):
                samples = samples.sample(NS)
            mu[clsi] = np.array(samples).mean(axis=0)

    for k in range(K):
        mu[k] /= mu[k].sum()

    fig = plt.figure(figsize=(16,K))
    for k in range(K):
        subplot = fig.add_subplot(K, N+1, k*(N+1)+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(mu[k].reshape(28,28), cmap=plt.cm.gray_r)
    fig.show()

    # N回のIterationを実施
    for iter_num in range(N):
        print("iter_num %d" % iter_num)

        # E phase
        resp = DataFrame()
        for index, line in df.iterrows():
            tmp = []
            for k in range(K):
                a = mix[k] * bern(line, mu[k])
                if a == 0:
                    tmp.append(0.0)
                else:
                    s = 0.0
                    for kk in range(K):
                        s += mix[kk] * bern(line, mu[kk])
                    tmp.append(a/s)
            resp = resp.append([tmp], ignore_index=True)

        # M phase
        mu = np.zeros((K, 28*28))
        for k in range(K):
            nk = resp[k].sum()
            mix[k] = nk/data_num
            for index, line in df.iterrows():
                mu[k] += line * resp[k][index]
            mu[k] /= nk

            subplot = fig.add_subplot(K, N+1, k*(N+1)+(iter_num+1)+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.imshow(mu[k].reshape(28,28), cmap=plt.cm.gray_r)
        fig.show()

    fig.savefig("mixem-figure_1-K{}-NS{}-{}.png".format(K, NS, time.asctime()[11:19]))

    # トレーニングセットの文字を分類
    cls = []
    for index, line in resp.iterrows():
        cls.append(np.argmax(line[0:]))

    print("Elapsed = {:.2f} sec.".format(time.time() - start_time))

    # 分類結果の表示
    show_figure(mu, cls)

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

