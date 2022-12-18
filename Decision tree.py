# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Function importing Dataset
def ReadData():
    data = pd.read_csv("BankNote_Authentication.csv")
    return data


def split(all, split_ratio):
    x = all.values[:, 0:4]
    y = all.values[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1 - split_ratio)
    return x_train, x_test, y_train, y_test


def train_tree(x_train, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf


def test_tree(x_test, clf):
    y_predict = clf.predict(x_test)
    return y_predict


def exp1(data):
    print("at exp1: ")

    for i in range(5):
        x_train, x_test, y_train, y_test = split(data, 0.25)
        clf = train_tree(x_train, y_train)
        y_predict = test_tree(x_test, clf)
        size = clf.tree_.node_count
        accuracy = accuracy_score(y_test, y_predict) * 100
        print(f"  test {i + 1} :")
        print("     Tree size = ", size)
        print("     Accuracy = ", accuracy)


def exp2(data):
    print("at exp2: ")

    mn_acc = 101
    mx_acc = -1
    mean_acc = 0
    mn_sz = 1000000
    mx_sz = -1
    mean_sz = 0

    training_size = []
    accuracy_lst = []
    nodes_lst = []
    split_ratio = 0.3

    for i in range(5):
        x_train, x_test, y_train, y_test = split(data, split_ratio)
        clf = train_tree(x_train, y_train)
        y_predict = test_tree(x_test, clf)

        accuracy = accuracy_score(y_test, y_predict) * 100
        mn_acc = min(mn_acc, accuracy)
        mx_acc = max(mx_acc, accuracy)
        mean_acc += accuracy

        size = clf.tree_.node_count
        mx_sz = max(size, mx_sz)
        mn_sz = min(mn_sz, size)
        mean_sz += size
        training_size.append(len(x_train))
        accuracy_lst.append(accuracy)
        nodes_lst.append(size)
        split_ratio += 0.1

    mean_acc /= 5
    mean_sz /= 5
    # print(f" test {i + 1} :")
    print("     max accuracy = ", mx_acc)
    print("     min accuracy = ", mn_acc)
    print("     Accuracy mean = ", mean_acc)
    print("     max size = ", mx_sz)
    print("     min size = ", mn_sz)
    print("     Tree size mean = ", mean_sz)


    plt.plot(training_size, accuracy_lst)
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(training_size, nodes_lst)
    plt.xlabel('Training set size')
    plt.ylabel('Number of nodes')
    plt.show()


def main():
    data = ReadData()
    exp1(data)
    exp2(data)


main()
