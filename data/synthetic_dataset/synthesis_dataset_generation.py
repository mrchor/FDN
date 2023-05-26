# -*- encoding: utf-8 -*-
"""
@File    : gen_data2.py
@Time    : 2022/4/18 19:30
@Author  : zhoujie
@Email   : zhoujiee@buaa.edu.cn
@Software: PyCharm
Randomly generate multi-task dataset
"""
from tqdm import tqdm
import scipy as sp
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection  import train_test_split
from sklearn.utils import shuffle


def data_preparation(sample_size = 10, num_dimension = 512, c = 1.0, m = 10):
    # Synthetic data parameters
    # Initialize vectors u1, u2, w1, and w2 according to the paper
    u1 = np.random.normal(size=num_dimension)
    u1 = (u1 - np.mean(u1)) / (np.std(u1) * np.sqrt(num_dimension))
    u2 = np.random.normal(size=num_dimension)
    u2 = (u2 - np.mean(u2)) / (np.std(u2) * np.sqrt(num_dimension))
    us = np.random.normal(size=num_dimension)
    us = (us - np.mean(us)) / (np.std(us) * np.sqrt(num_dimension))
    w1 = c * u1
    w2 = c * u2
    ws = c * us

    # Feature and label generation
    alpha = np.random.normal(size=m)
    beta = np.random.normal(size=m)
    delta = np.random.normal(size=num_dimension)
    gamma = np.random.normal(size=num_dimension)
    X = []
    labels = []

    for i in tqdm(range(sample_size)):
        x1 = np.random.normal(loc=-1.0, scale=1.0, size=num_dimension)
        x2 = np.random.normal(loc=1.0, scale=1.0, size=num_dimension)
        xs = np.random.normal(loc=0.0, scale=1.0, size=num_dimension)

        # generate features
        epsilon = np.random.normal(size=1)[0]
        X_1 = [np.sin(delta[i] * (u1[i] * x1[i] + us[i] * xs[i]) + gamma[i])
               + np.cos(delta[i] * (u1[i] * x1[i] + us[i] * xs[i]) + gamma[i])
               + epsilon for i in range(num_dimension)]
        X_2 = [np.sin(delta[i] * (u2[i] * x2[i] + us[i] * xs[i]) + gamma[i])
               + np.cos(delta[i] * (u2[i] * x2[i] + us[i] * xs[i]) + gamma[i])
               + epsilon for i in range(num_dimension)]
        X_s = [np.sin(delta[i] * (u1[i] * x1[i] + u2[i] * x2[i] + us[i] * xs[i]) + gamma[i])
               + np.cos(delta[i] * (u1[i] * x1[i] + u2[i] * x2[i] + us[i] * xs[i]) + gamma[i])
               + epsilon for i in range(num_dimension)]
        X.append(np.array(X_1 + X_2 + X_s))

        # generate label
        # shared
        y1 = ws.dot(xs) + w1.dot(x1)
        y2 = ws.dot(xs) + w2.dot(x2)
        comp = 0.0
        for j in range(m):
            comp += np.sin(alpha[j] * ws.dot(xs) + beta[j])
        y1 += comp
        y2 += comp

        # specific
        comp1, comp2 = 0.0, 0.0
        for j in range(m):
            comp1 += np.sin(alpha[j] * w1.dot(x1) + beta[j])
            comp2 += np.sin(alpha[j] * w2.dot(x2) + beta[j])

        y1 += comp1
        y1 += np.random.normal(scale=0.01, size=1)[0]
        y2 += comp2
        y2 += np.random.normal(scale=0.01, size=1)[0]

        labels.append([y1, y2])
    X = np.array(X)
    features = pd.DataFrame(X)
    labels = pd.DataFrame(labels)
    samples = pd.concat([labels, features], axis=1)
    # random
    samples = shuffle(samples)
    train_data, test_data = train_test_split(samples, test_size=0.3, random_state=2022)
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--sample_size', type=int, default=10000)
    args = parser.parse_args()
    print("data generating...")
    train_data, test_data = data_preparation(sample_size=args.sample_size, num_dimension=512)
    train_data.to_csv('./train_data/train_data.csv', header=None, index=None)
    test_data.to_csv('./test_data/test_data.csv', header=None, index=None)
    print('data generate finish !')

if __name__ == "__main__":
    main()


