# 超参数调节、Batch正则化、编程框架（Hyperparameter tuning、Batch Normalization、Programming Frameworks)

***
### 如何选择合适的超参数？

在之前的学习中，我们接触到了越来越多的超参数，这些超参数是决定我们训练速度和质量的基础，我先简单列一下我们接触到的超参数们：
- learning-rate α
- momentum参数  β
- adam参数 β1，β2，ε
- 网络层数 layers
- 每一层神经元个数 hidden-units
- 学习速率下降系数
- mini-batch size

这么多的超参数需要我们通过特定的领域知识，甚至是交叉训练来一一确定，这些参数的重要性不一定相同，譬如learning-rate可能就比较重要，mini-batch siz可能相对来说就不那么重要。
因为参数之间的不对等性，我们在寻找参数时不能进行方格等距搜索而要进行随机的搜索，意思是像下面这样的方式是不可取的：
![](https://i.imgur.com/vd7PyW4.png)
而要在整个方块内随机取点，这样我们可以得到更多的参数组合，从而更可能发现最好的参数。
其次，选择一个恰当的参数搜索范围也是至关重要的。假设一个这样的例子：一个参数α的合理范围是0.0001到1，但可能越接近0对结果影响越大，参数越敏感，那么我们就不能让它在（0.0001，1）这个区间上随机取值了，因为这样的话90%的取值都落在了（0.1，1）上，反而越敏感的区间越娶不到几个值了。所以应该根据参数的敏感程度划分随机的区间，譬如让其在（0.0001，0.001）（0.001，0.01）（0.01，0.1）（0.1，1）这四段上等概率取值即可。
***

### Batch正则化
之前我们说过对输入的数据进行正则化处理可以加快学习速率，现在我们考虑每一层的输入，如果把每一层拿出来单独看的话，它的输入就是上次层的输出，那么把这个输入正则化有没有作用呢？答案是有的，下面我们来看看如何对每一层的输出进行正则化。类似于对输入的正则化，我们需要求出每个数据经过线性运算后z的均值和方差：
![](https://i.imgur.com/IITdD1H.gif)
![](https://i.imgur.com/jCxfvVz.gif)
然后利用均值和方差进行正则化输出：
![](https://i.imgur.com/Oitwb0X.gif)
这样就可以得到均值为0，方差为1的标准输出，这里面的epsilon是为防止除0错而加的一个非常小的值。
最后我们可以再根据需要设置需要的均值和方差；
![](https://i.imgur.com/3L8ni0s.gif)
最后得到的这个z一弯就是传进下一步激活函数的输入了。
总体来看，这样做让一层的所有神经元输出值有了一个相同的
均值和方差。
因为我们有将所有z均值归零这一步，所以实际上b参数就不需要了，对于每一层我们新的参数是：w、beta、gamma。对这三个参数可以进行梯度下降，或者adam等方法来训练。
通过这样的正则化可以加快训练速度这是显而易见的，但这个batch正则化还有附带的功能，那就是它还有轻微的避免过拟合的作用，类似droupout，怎么理解呢?
通过这样的batch操作，一层的输出对下一层的输入的影响被弱化了，这就使得单个神经元不会对某一个输入过分敏感，从而一定程度上减小过拟合，但这种效果是有限的。
还有一点需要注意：就是在预测时好像出现了些问题，我们要正向传播要求方差和均值，但1个数据怎么求呢？这就需要我们拿训练集的均值和方差来近似了，整体的复杂度有不小的提高。其实际效果只能因不同的模型来定。

***
### 多分类与Softmax回归
之前我们的神经网络输出是0或1，代表两个类别，现在我们要将其拓展到多分类上，只需要该表最后的输出层即可。
回想之前的逻辑回归，我们最后输出的是一个0，1之间的值，代表属于这一类的概率，那么如果有C类的话，我们就要输出C个0，1之间的值，分别代表属于其中一类的概率，而且相加为1。怎么做呢？
我们改动一下最后的输出层的激活函数即可。假如说有4类，那么我们的输出层就应该接受到4个输入，运算后得到的z应该是一个4*1的列向量，譬如【5，-1，3，1】然后我们的激活函数就是对这个向量先取e再将每一项除以他们的和，即变为：
![](https://i.imgur.com/T8yG8jl.gif)
这样就变为了没一类相应的概率。
那么损失函数怎么定义呢？模仿之前逻辑回归的损失函数不难得到：
![](https://i.imgur.com/ycR2Bpc.gif)
这里的y向量是真实类别为1，其余项为0的C*1向量。不难验证这个式子是对的。
那反向传播呢？
这里就不一步一步推导了，直接给出结果：
![](https://i.imgur.com/1mqJvnl.gif)
有兴趣的可以自己推一下。
好啦，剩下的就是按照传统的梯度下降或其他方法来啦，是不是很简单呢

***
### 深度学习框架，tensorflow

学了这么多关于神经网络的方法难道我们解决一个问题要从头开始实现吗？当然不必，有很多已经实现好的深度学习的框架来帮助我们快速的搭建一个属于自己的神经网络，现在能接触到的库有很多，处理不同的问题可能需要不同的工具，这里简单列几个常用的库：
- Caffe/Caffe2
- CNTK
- DL4J
- Keras
- Lasagne
- mxnet
- PaddlePaddle
- TensorFlow
- Theano
- Torch

这些框架都有官方的说明文档来告诉你该怎么用，这里详细介绍一下其中最常用的一个：TensorFlow

在TensorFlow中你需要做的就是把损失函数的形式写出来，然后它就能自己找到反向传播的方法，十分智能。
下面的代码是一个损失函数为w^2-20W+25的网络的训练过程：
```python
import numpy as np
import tensorflow as tf

coeddicients = np.array([[1],[-20],[25]])

w=tf.Varible([0],dtype=tf.float32)
x=tf.placeholder(tf.float32,[3,1])
cost = =x[0][0]*w**2+x[1][0]*w+x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
for i in range(1000):
	session.run(train,feed_dict={x:cofficients})
print(session.run(w))
```
最后输出的结果是w为9.999998，基本上达到了最低点。

这个框架还是非常好用的，详细用法参见代码或官方文档。

拜拜！！