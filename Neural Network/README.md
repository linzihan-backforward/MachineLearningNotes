# 神经网络（Neural Network）

### 逻辑回归的延伸
上一节我们看到运用逻辑回归的模型，我们可以训练出一个还可以的二分类分类器，但是其也存在很多问题，如：测试集准确度不高，过拟合，分类局限等。这一节中我们考虑来进一步优化模型。在逻辑回归中，我们是用一个线性函数来进行分类，线性函数本身复杂程度比较低，对于复杂模型不能很好的预测，我们可以通过进行多次的类逻辑回归来设计一个更复杂的模型，而这个模型就是神经网络了。

***
### 初识神经网络

神经网络这个概念最初是指生物学中的大脑中神经元组成的庞大的神经信号传导网络，就像下面这个图:
![](https://i.imgur.com/tKN8FcX.jpg)
生物中的神经网络其信号传导大概是一个神经元细胞与多个细胞相连，多个细胞都可以对其释放神经递质，也就是化学信号，而此神经元细胞在收到信号后，可以对多个信号综合处理从而得出对下一个细胞释放什么样的信号。
那么这里，在数学或机器学习中的神经网络也具有相似的特点，一个两层的神经网络可以用下面这张图形象地表示。
![](https://i.imgur.com/guQwXGV.png)
如右边这个图，我们的输入值为x1，x2，x3，输出的值就是预测值，箭头表示了值的传递，每个圆圈代表一个神经元。而左边的图就是每个神经元放大后的情况，我们可以看到，在我们这个模型中每个神经元接收三个输入，然后对这个输入进行一次逻辑回归，得出的就是输出值。是不是感觉很熟悉？其本质就是进行两次Logistic回归嘛！

***
### 参数向量化与梯度传播
按照我们上图中的模型，我们有4个神经元也就有4个W参数和4个b参数，回想上一节中的情况，上一节中我们的W是一个列向量，b是一个实数，而现在我们需要求4个列向量，4个实数，是不是可以将这些参数再进一步合并为一个矩阵和一个列向量呢？答案是可以的。证明很容易，这里省略。
那么最后有这样一个结论：对于m个样本，每个样本n个特征的两层神经网络，我们第一层的W参数是一个n行m列的矩阵，b是一个m行1列的列向量，第二列测参数通逻辑回归一样。这样我们的这个预测值的计算过程就可以用矩阵运算一次得出。像这样：
```python
Z1=np.dot(W1.T,X)+b1
A1=sigmoid(Z1)
Z2=np.dot(W2.T,A1)+b2
A2=sigmoid(Z2)
```
这样最后的A2就是一个包含m个预测值的列向量。

掌握了正向传播的方法，我们再看看如何进行反向传播，显然，我们的计算过程可以用这个图表示：
![](https://i.imgur.com/gurx1ty.png)
在这个过程中我们要求4个偏导，因为这里的第二层就是一个传统的逻辑回归，所以我们在逻辑回归中得出的结论依旧适用
![](https://i.imgur.com/n0BXUr8.gif)

![](https://i.imgur.com/LnpoW5r.gif)

![](https://i.imgur.com/6igiuqx.gif)

关键是我们从第二层反推到第一层时产生的变化，利用链式求导法则我们可以得出这样的结果：
![](https://i.imgur.com/oesyKmx.gif)
这样我们对于梯度的通用形式又可以得到适用，如下：
![](https://i.imgur.com/v940uhV.gif)

![](https://i.imgur.com/byGY8lz.gif)
这样我们只需要在上一节的逻辑回归中再添几步，就可以升级为一个简单神经网络的模型了。
```python
Z1=np.dot(W1.T,X)+b1
A1=sigmoid(Z1)
Z2=np.dot(W2.T,A1)+b2
A2=sigmoid(Z2)
dZ2=A2-Y
dW2=np.dot(dZ2,A1.T)/m
db2=np.sum(dZ2)/m
dZ1=np.dot(W2,dZ2)*(A1)*(1-A1)
dW1=np.dot(dZ1,X.T)/m
db1=np.sum(dZ1)/m
```
是不是和逻辑回归统一起来了呢？其本质上就是用高维的W而非一维的W来就行逻辑回归。

***
### 激活函数的选择与非线性

在逻辑回归中我们的sigmoid函数是用来干啥的呢？好像是用来让预测值落在0，1之间的，对于二分类问题，我们需要标准化结果，所以需要这样的函数，那么如果我们的预测值空间是全体实数的话是不是就可以去掉这个sigmoid函数了呢？
答案是，在逻辑回归中可以，在神经网络中不行！因为如果去掉的话不管网络有多少层都会退化成和逻辑回归相同的模型。如果你有兴趣的话可以拿我们前面列的正向传播的式子推一下，最后会发现几个参数矩阵会乘到一起，成为一个参数矩阵，则就变成了逻辑回归。
那么这个激活函数只有sigmoid函数吗？答案是不是，而且恰恰很少用sigmoid函数，sigmoid函数虽然可以标准化预测值但是其在远离原点时梯度太小，导致学习速度很慢。
常用的计划函数包括这几个：
![](https://i.imgur.com/67RqL5H.png)
双曲正切函数tanh，这个函数可以使输出值保持对称性，这种对称性在多层网络中是十分重要的。
![](https://i.imgur.com/AuQVzQK.png)
max（0，z）最大值函数，又成ReLU（Rectified Linear Units）这个函数在大于0时有一个恒定的梯度1，这种性质可以加快学习速度
![](https://i.imgur.com/8frpABl.png)
max（0.01z,z）ReLU函数中如果小于零的话会出现梯度消失的现象，这将使得我们无法完成学习，为了克服这个缺点，又引入了这个Leaky ReLU

**这其中最常用的是ReLU**

***
### 随机初始化

在逻辑回归中初始化似乎不是个问题，只需将所有参数都置0即可，但在神经网络中我们不能这么做，回想一下最早的那张抽象图，我们第二层有3个神经元，按理来说这3个神经元应该是不一样的，即W应不同，如果我们将其全部初始化为0，那么我们会发现由于对称性，其反向传播一遍的梯度也一样，则不管怎么学习，这三个W都是一样的，那么我们的三个神经元不就退化成1个了吗，所以我们不能初始化成一样的。而b就不存在这个问题，如果W不一样那么对称性消失，b的梯度自然也就不一样了。
在numpy中可以这样来初始化一个随机矩阵
`W1=np.random.randn((n,m))`
`b1=np.zeros((m,1))`
`w2=np.random.randn((m,1))`
`b2=0`

***
好啦，这一节的神经网络就到这里，是不是感觉上路了呢？后面还有更精彩的内容呢！
（详细内容请看代码）