import tensorflow as tf
import pandas as pd

# 设置日志
tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_data_path = "iris_training.csv"
test_data_path = "iris_test.csv"

# 预测数据
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
expected = ['Setosa', 'Versicolor', 'Virginica']


# 定义输入函数
def input_data(features, labels, training=True, batch_size=256):
    # 将输入转换为数据集。
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # 如果在训练模式下混淆并重复数据。
    if training:
        tf_dataset = tf_dataset.shuffle(1000).repeat()
    return tf_dataset.batch(batch_size)


def load_data():
    train = pd.read_csv(train_data_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_data_path, names=CSV_COLUMN_NAMES, header=0)
    return train, test


def feature_columns(train):
    featureColumns = []
    for key in train.keys():
        featureColumns.append(tf.feature_column.numeric_column(key=key))
    return featureColumns


def input_fn(predict_x, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(predict_x)).batch(batch_size)


def train_and_test():
    train, test = load_data()
    train_y = train.pop("Species")
    test_y = test.pop("Species")
    train.head()
    feature_columns_slices = feature_columns(train)
    # 构建模型 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络。
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns_slices,
                                            hidden_units=[30, 10],  # 隐藏节点分别为 30 和 10
                                            n_classes=3)  # 模型必须从三个类别中做出选择。
    # 训练模型 classifier 为一个 Estimator 实例
    classifier.train(input_fn=lambda: input_data(train, train_y, training=True),
                     steps=5000)

    # 评估经过训练的模型。
    eval_result = classifier.evaluate(input_fn=lambda: input_data(test, test_y, training=False))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 预测
    predictions = classifier.predict(input_fn=lambda: input_fn(predict_x))

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))


if __name__ == '__main__':
    train_and_test()
