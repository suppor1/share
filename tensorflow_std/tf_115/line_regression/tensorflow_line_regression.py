import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# 线性回归

# 读取特征
def get_feature(df):
    print("read feature===========================")
    ones = pd.DataFrame({"ones": np.ones(len(df))})  # 创建 m行 1列的dataframe
    ones.head()
    ones.info()
    # 合并数据，根据列合并
    data = pd.concat([ones, df], axis=1)
    print(type(data))
    return data.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵, as_matrix() 方法已经过期，使用 values 代替


# 读取标签列
def get_label(df):
    return df.iloc[:, -1].values


# 特征缩放
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


# 代价函数
def lr_cost(theta, X, y):
    m = X.shape[0]  # m 为 样本数
    inner = X.dot(theta) - y
    square_sum = inner.T.dot(inner)
    cost = square_sum / (2 * m)
    return cost


# batch gradient decent 批了梯度下降
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T.dot(X.dot(theta) - y)
    return inner / m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    """
    拟合线性回归，返回参数和代价
    :param theta:
    :param X:
    :param y:
    :param epoch: 批处理的轮数
    :param alpha:
    :return:
    """
    cost_list = [lr_cost(theta, X, y)]
    _theta = theta.copy()

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_list.append(lr_cost(_theta, X, y))

    return _theta, cost_list


def view_cost(cost_list, epoch):
    """
    代价函数可视化
    :param cost_list:
    :param epoch:
    :return:
    """
    dic = {"epoch": list(range(0, epoch + 1)), "cost": cost_list}
    df = pd.DataFrame(dic)
    ax = sn.lmplot("epoch", "cost", df, fit_reg=False)
    plt.show()


def line_regression_std():
    # 读文件， 设置列名
    df = pd.read_csv("ex1data1.txt", names=["population", "profit"])
    # 显示前 5 行
    print(df.head())
    # 显示 info
    print(df.info)
    # 看下全量数据
    sns = sn.lmplot('population', 'profit', df, size=10, fit_reg=False)
    # plt.show()
    feature = get_feature(df)
    print("feature info.....")
    print(feature.shape, type(feature))
    print(feature)
    label = get_label(df)
    print("label info.....")
    print(label.shape, type(label))
    print(label)
    normalize = normalize_feature(df)
    print(type(normalize))
    print(normalize)
    theta = np.zeros(feature.shape[1])
    print("theta:", theta)
    cost = lr_cost(theta, feature, label)
    print("cost:", cost)
    epoch = 500
    final_theta, cost_list = batch_gradient_decent(theta, feature, label, epoch)
    print("final_theta", final_theta)
    print("cost")
    for _cost in cost_list:
        print(_cost)
    # 代价函数可视化
    view_cost(cost_list, epoch)
    # y = theta1 * x + theta0 * 1
    b = final_theta[0]  # y轴的截距
    m = final_theta[1]  # 斜率
    plt.scatter(df.population, df.profit, label="Training data")
    plt.plot(df.population, df.population * m + b, label="Prediction")
    plt.legend(loc=2)
    plt.show()


# 正规方程求解
def normalEqn(feature, label):
    theta = np.linalg.inv(feature.T.dot(feature)).dot(feature.T).dot(label)
    return theta


def linear_regression(feature, label, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):
    X = tf.compat.v1.placeholder(tf.float32, shape=feature.shape)
    Y = tf.compat.v1.placeholder(tf.float32, shape=label.shape)

    with tf.compat.v1.variable_scope("linear-regression"):
        W = tf.compat.v1.get_variable("weight", shape=(X.shape[1], 1), initializer=tf.constant_initializer())
        y_predict = tf.matmul(X, W)
        loss = 1.0 / (2 * len(feature)) * tf.matmul((y_predict - Y), (y_predict - Y), transpose_a=True)

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    loss_data = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: feature, Y: label})
            loss_data.append(loss_val[0, 0])
            if len(loss_val) > 1 and np.abs(loss_val[-1] - loss_val[-2]) < 10 ** -9:
                break

    tf.compat.v1.reset_default_graph()
    return {"loss": loss_data, 'parameters': W_val}


if __name__ == '__main__':
    # line_regression_std()

    #  一下为选修
    data = pd.read_csv("ex1data2.txt", names=["square", "bedrooms", "price"])
    print(data.head())
    data = normalize_feature(data)
    print(data.head())
    print(data.shape)
    feature = get_feature(data)
    print(feature.shape, type(feature))
    label = get_label(data)
    print(label.shape, type(label))
    alpha = 0.01  # 学习率
    theta = np.zeros(feature.shape[-1])
    print(theta)
    epoch = 500
    final_theta, cost_data = batch_gradient_decent(theta, feature, label, epoch, alpha)
    print(final_theta)
    for _cost in cost_data:
        print(_cost)

    c = {"epoch": list(range(0, epoch + 1)), "cost": cost_data}
    data_df = pd.DataFrame(c)
    sn.lmplot("epoch", "cost", data_df, fit_reg=False)
    # plt.show()

    print(final_theta)
    normal_theta = normalEqn(feature, label)
    print(normal_theta)

    base = np.logspace(-1, -5, num=4)
    print(base)
    print("=========================================")
    candidate = np.sort(np.concatenate((base, base * 3)))
    print(candidate)
    epoch = 50
    fig, ax = plt.subplots(figsize=(16, 9))
    for alpha in candidate:
        _, cost_data = batch_gradient_decent(theta, feature, label, epoch, alpha=alpha)
        ax.plot(np.arange(epoch + 1), cost_data, label=alpha)
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('cost', fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title('learning rate', fontsize=18)
    # plt.show()

    # run the tensorflow graph over several optimizer
    label = label.reshape(len(feature), 1)
    print(label.shape, type(label))
    print(feature.shape, type(feature))
    epoch = 2000
    alpha = 0.01
    optimizer_dict = {"GD": tf.compat.v1.train.GradientDescentOptimizer,
                      "Adagrad": tf.compat.v1.train.AdagradOptimizer,
                      "Adam": tf.compat.v1.train.AdamOptimizer,
                      "Ftrl": tf.compat.v1.train.FtrlOptimizer,
                      "RMS": tf.compat.v1.train.RMSPropOptimizer}

    results = []
    for name in optimizer_dict:
        res = linear_regression(feature, label, alpha, epoch, optimizer_dict[name])
        res["name"] = name
        results.append(res)

    fig, ax = plt.subplots(figsize=(16, 9))

    for res in results:
        loss_data = res['loss']

        #     print('for optimizer {}'.format(res['name']))
        #     print('final parameters\n', res['parameters'])
        #     print('final loss={}\n'.format(loss_data[-1]))
        ax.plot(np.arange(len(loss_data)), loss_data, label=res['name'])

    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('cost', fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title('different optimizer', fontsize=18)
    plt.show()
