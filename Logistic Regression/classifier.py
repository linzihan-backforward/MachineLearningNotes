import numpy as  np


# 这里我们用train_set_x代表训练集图片，train_set_y代表训练集标签，0代表非猫，1代表是猫。test_set_x代表我们的测试集
# train_set_x是一个二维数组，每一列代表一个图片数据，所有3维像素值被拉成一列，并统一除以了255以实现标准化
# train_set_y是一个列向量，代表每个数据的标签

# 利用numpy的广播功能实现我们的sigmoid函数
def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a


# 初始化w为dim行1列的列向量，b为实数0
def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

# 前向传播计算损失值，后向传播计算梯度
# w，b是模型参数，X为数据集，Y为标签值
def propagate(w, b, X, Y):
    # m是数据的个数
    m = X.shape[1]

    #前向传播
    # A是预测值，cost是损失值
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -(np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T)) / m  # compute cost

    #后向传播
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m


    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

# 参数训练函数
# w,b,X,Y含义参上，num_iterations是训练次数，learn_rate是每一步的步长，print_cost为True代表每100次训练打印一次当前损失
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):


        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        #参数更新
        w = w - learning_rate * dw
        b = b - learning_rate * db


        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}
    #返回两个字典代表最终的w,b和最终的梯度，一个list为每次训练的损失值
    return params, grads, costs

# 预测函数
# w,b是已经训练好的参数，X是要预测的数据集
def predict(w, b, X):


    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    #将w变为列向量
    w = w.reshape(X.shape[0], 1)


    A = sigmoid(np.dot(w.T, X) + b)


    for i in range(A.shape[1]):
        if (A[0][i] <= 0.5):
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

