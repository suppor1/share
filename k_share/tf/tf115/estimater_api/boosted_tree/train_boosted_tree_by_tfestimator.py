import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt

tf.random.set_seed(123)

feature_columns = []


def train_boosted_tree():
    # 加载数据集。
    dftrain, dfeval, y_train, y_eval = load_data()
    print(dftrain.head(2))
    print(dftrain.describe())
    print(dftrain.shape)
    print(dfeval.shape)
    print(dftrain['sex'].unique())
    # dftrain.age.hist(bins=30, rwidth=1.1)

    # dftrain.sex.value_counts().plot(kind='barh')

    # dftrain['class'].value_counts().plot(kind='barh')

    # dftrain['embark_town'].value_counts().plot(kind='barh')

    # pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    # plt.show()
    # 创建特征列与输入函数
    tc = tf.feature_column
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']

    for column_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[column_name].unique()
        oneHot = one_hot_cat_columns(column_name, vocabulary)
        feature_columns.append(oneHot)

    for column_name in NUMERIC_COLUMNS:
        number_column = tf.feature_column.numeric_column(column_name, dtype=tf.float32)
        feature_columns.append(number_column)

    print("len: ", len(feature_columns))
    print(feature_columns)
    # 可以查看特征列生成的转换 使用 tf.feature_columns
    test_columns(dftrain)


def test_columns(dftrain):
    example = dict(dftrain.head(1))
    print("可以查看特征列生成的转换 使用 tf.feature_columns")
    print(dftrain.head(1))
    print(example)


def one_hot_cat_columns(column_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(column_name, vocab)
    )


def load_data():
    dftrain = pd.read_csv('./train.csv')
    dfeval = pd.read_csv('./eval.csv')
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')
    return dftrain, dfeval, y_train, y_eval


if __name__ == '__main__':
    train_boosted_tree()
