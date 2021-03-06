# 深层神经网络(Deep Neural Network)
***
通过上一节，我们已经对神经网络有了一个简单的认识，我们训练了一个两层的神经网络，并发现了其比简单逻辑回归更出彩的地方，但是这个模型还是不够复杂，对维度更高的关系无法提供准确的预测，那么就需要引出深层的神经网络，现在深层神经网络已经成为深度学习领域的主流，在各种模型预测上都呈现出优异的效果。
***
### 初识深层神经网络
类比我们上节中从逻辑回归拓展到神经网络的方法，我们维持输入层和输出层不变，而增多我们隐藏层的数量，隐藏层越多，我们就说网络越深，如下面这张图所示
![](https://i.imgur.com/xemCKow.png)
其中每个节点就是一个神经元，都有各自的激活函数与参数矩阵。最终的输出层保持一个输出神经元从而得到我们的预测值。我们将输入层记为第0层，那么我们整个网络的传播可以写成以下形式：![](https://i.imgur.com/QGvMISa.gif)
即每一层都将上一层的输出作为新的输入来学习新的参数，从而最终得到一个复杂的传播网络。

***
### 为何深层网络效果好？

有人可能这样的困惑，为什么我们要通过加深网络来获得更好的预测精度而不是通过增多我们单层的神经元个数呢？
这其中其实有很深的原理我现在也无法理解，只能先从两个简单的方面说一下：
**1.深层的网络可以更好的处理复杂的特征。**

在可视化的神经网络如卷积神经网络中，我们可以发现这样一个规律：浅层的神经元会对数据中的某些基本项敏感度更高，如图片中物体的边界，而深层的神经元会通过对这些基本项的组合来识别更复杂的数据，如人的眼睛、耳朵等。这与生物学中人脑神经元的工作模式很相近，那么要迎合复杂的特征，自然我们要加深网络的深度。

**2.对同一个功能深层网络比浅层网络更省神经元**

这点在电路逻辑中有很强的体现，在此不展开讲。举个例子：如果要完成n个数的异或运算，深层网络中需要logn个神经元如二叉树般运算即可，但浅层网络就必须要有指数级的神经元才能完成这个运算，所以我们用深层网络就可以以更小的参数规模完成指定任务。

***
### 模块化深层神经网络
在浅层神经网络中我们可以通过加几行代码来完成前向和后向传播，但如果层数多的话这样就会显得不够清晰。所以我们尝试将计算过程模块化。
![](https://i.imgur.com/hReN6fL.png)
如上图，我们可以将每一层作为一个模块，前向传播如第一行所示，每个模块需要前一个模块的A作为输入，自身成员包括W和b，然后输出下一个A给下一个模块。
这其中我们要把每个模块的Z记录下来以用于反向传播。
反向传播类似，每个模块接受一个dA作为输入，自身成员包括W，b、dZ以及存下来的Z，输出为下一个dA，同时计算出dW和db来完成梯度下降。
这里每个模块的代码都与之前的类似，这里就不写伪码了。

***
### 参数与超参数
什么是参数？
我们的神经网络中每个神经元所拥有的W和b是参数。
什么是超参数？
超参数就是确定这个网络的其余参数如：网络层数、每层神经元个数、每个神经元的激活函数、学习速度、迭代次数。

超参数是我们完成算法所必须事先确定的，这其中的很多选择都直接决定了我们模型预测的结果，而我们是没有办法知道什么样的超参数适合什么样的问题的，所以只能在不断的尝试和调整中找到相对适合的。这也启示我们在机器学习中要多改变、多调整，没准就可以优化结果。

***
### 神经网络与人脑

我们人脑的神经系统是一个极其复杂的系统，神经网络只是其的一个简化模型，并不能说神经网络就是人脑的工作模式，对人脑的研究依旧是现在最前沿的课题之一，神经网络可以说是在数学领域模拟人脑比较成功的一个案例罢了。
***
好了，我们已经掌握了神经网络的模型了，其他各种形式的神经网络均是在此原理上调整与发展而来的。


 