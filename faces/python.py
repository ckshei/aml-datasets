import numpy as np
from scipy import misc
import csv

train_labels, train_data = [], []
test_labels, test_data = [], []
for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

for line in open('./faces/test.txt'):
    im = misc.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])
test_data, test_labels = np.array(test_data, dtype=float), np.array(test_labels, dtype=int)

np.savetxt("train_labels.csv", train_labels, delimiter=",")
np.savetxt("train_data.csv", train_data, delimiter=",")
np.savetxt("test_labels.csv", test_labels, delimiter=",")
np.savetxt("test_data.csv", test_data, delimiter=",")