# 机器学习笔记——————逻辑回归（Logistic Regression）

***
### 二分类问题
现在假设我们有一对（x,y）,其中x是一个实数，y是0或1。代表x是或不是这一类，比如(7,1)代表7是质数，（10，0）代表10不是质数。现在假设我已经有了一堆（x，y）对，即训练数据。要求对一个测试的x建立一个可靠的模型使得可以预测出y来。

***
### 线性回归
我们建立一个线性的模型来进行拟合。即\\( f(x)={w*x+b}\\) 
但是这样算出来的`f(x)`是一个值域为`R`的实数，我们要将它映射到（0，1）上。
这时我们用到了一个函数叫Sigmoid函数 ,其表达式为$$\sigma(x)={1\over1+e^{-x}}$$
图像如下（自Baidu）：
![图片无法显示](https://i.imgur.com/BdEw4Qj.png)
这个函数定义域为全体实数，值域为(0,1),且当|x|>5时就基本收敛，但因为其毕竟不是只能取0、1两个值的函数，所以我们求出来的\\(\widehat y\\)也不是最终的分类结果，而是概率，即f(x)等于x属于1这一类的概率。

***
### 损失函数（Loss Function）
为了衡量我们的f(x)的准确程度，我们要想办法找到其与真实分类的差值，设我们的预测概率为\\(\widehat y\\)，真实分类为y，运用极大似然估计的方法，可以得到下面的函数：
$$ l(y,\widehat y)={\widehat y^y(1-\widehat y)^{1-y}}$$
这个函数将两类统一起来，我们可以看到，不论y取0还是1，我们都希望这个\\({l(y,\widehat y)}\\)取1，所以损失函数找到了，就是这个l与1的差值，但因为指数函数的一些不好的性质，我们习惯将其取log，定义Logistic Regression的损失函数如下：
$$ l(y,\widehat y)={-(yln\widehat y+(1-y)ln(1-\widehat y))}$$

这是对一个数据x的损失，我们要将给的m个数据都利用起来，很容易想到的就是取平均值啊，我们也确实是这个做的，对于全部数据集的损失函数如下：
$$L(w,b)={\sum_{i=1}^{m}l(w*x_i+b,y)}$$
现在我们找到了一个衡量我们预测模型的方法，现在要做的就是使这个损失函数最小。
找到了最小函数所对应的(w,b),我们就找到了基于我们逻辑回归的最优预测模型。

***
### 损失函数的梯度(Gradient)

我们高数中学过梯度的概念，这里我们简单复习一下，对于一个多元函数\\(f(x_1,x_2...x_n)\\)其梯度是一个n维向量，表示这个n维空间中函数f上升最快的方向，设梯度为\\(D={(dx_1,dx_2...dx_n)}\\)其中\\(dx_i={\partial f \over \partial x_i}\\)
也就是说，对于一个n维空间函数中的任意一点，只要沿着它的梯度方向的反方向走一步，就一定会走到一个至少不比原来点大的点。
所以对于上面的损失函数，我们只要一直沿着梯度的反方向走，就能找到最小值点。为了方便以后的运算，这里我们定义这样两个值$$\text{d}w={\partial L(w,b) \over \partial w} $$
$$\text{d}b={\partial L(w,b) \over \partial w} $$
如此一来，要找到损失函数的最小点，我们只需从一个随意的（w,b）开始，不断地进行$$b={b-\alpha \times \text{d}b} $$
$$w={w-\alpha \times  \text{d}w} $$
就一定可以到达一个局部最优点，又因为我们定义的损失函数是全局凸的（这就是为什么我们的损失函数不定义为欧式距离的原因），所以必然也就是全局最优点。
上式中的\\(\alpha\\)是一个自己设置的步长，代表每次向梯度反方向走多远。
现在我们要做的就是求梯度啦！

***
### 计算图中的链导法则

w
 \
x-------> z=w*x+b -----> \\(\widehat y=a={\sigma(z)}\\)----->\\(L(\widehat y,y)\\)
 /
b
根据高数里的链式求导法则，我们可以求出损失函数对于w和b的偏导
$${\partial L \over \partial w}={{\partial L \over \partial a}\times{\partial a \over \partial z}\times{\partial z \over \partial w}}$$
其中 \\({\partial L \over \partial a}=-{y\over a}+{1-y \over 1-a}\\)                            \\({\partial a\over \partial z}={\sigma(z)(1-\sigma(z))}\\)         \\({\partial z \over \partial w}=x\\)
将上面的乘起来就可以得到最终的梯度了
$${\partial L \over \partial w} ={(\sigma(w\times x+b)-y)\times x}$$
同理可得到:
$${\partial L \over \partial b}={\sigma(w\times x+b)-y}$$

***
### 逻辑回归整体流程

求出了梯度我们就可以正式的开始训练我们的模型了
先写一个简单的伪代码
```
L=0;dw=0;db=0; w=0;b=0;
for i=1 to m:
	zi=w*xi+b
    ai=sigmoid(zi)
    L+=-(yi*log(ai)+(1-yi)*log(1-ai))
    dzi=ai-yi
    dw+=xi*dzi
    db+=dzi
L/=m;dw/=m;db/=m;
```
上面就是进行参数初始化和一次训练的过程，一次训练完之后我们执行`w-=alpha*dw;b-=alpha*db;`就可以完成参数的更新，至于训练到什么程度，你可以选择到函数收敛，或者固定1000次什么的，视具体情况而定。

***
### 多维数据域的拓展与向量化

以上我们讨论的情况都是x是一个实数的情况，但是现实中我们处理的问题往往很复杂，需要分类的x都是一个包含n个实数的向量\\(X=(x_1,x_2,...x_n)\\),对于这种情况我们的整体思路和模型是没有变的，相对应的我们的W和b也都升级成为一个n维的向量。
我们代码中的数乘运算*相应的变为向量点乘运算。
这里我们使用python中numpy库，这个库可以实现强大的向量与矩阵运算，一些numpy的用法以后再整理。
改写上面的代码如下：
```python
import numpy as np
L=0;db=0;dw=np.zero((n,1)); w=np.zero((n,1));b=0;
for i=1 to m:
	zi=np.dot(w.T,xi)+b
    ai=sigmoid(zi)
    L+=-(yi*log(ai)+(1-yi)*log(1-ai))
    dzi=ai-yi
    dw+=xi*dzi
    db+=dzi
L/=m;dw/=m;db/=m;

```
这样就可以了，是不是感觉没什么变化？多亏了numpy中的广播机制使得我们操作向量跟操作一个数一样方便。

***
### 并行处理与矩阵化

做到上面那样按说就可以运行了，但是因为有时候数据集非常大，进行一遍循环非常耗时，我们想要进行并行处理，而最简单的方法就是将m个列向量合并成一个n×m的矩阵，python中的矩阵运算就是采用并行加速的。
现在我们尝试修改上面的代码，将最外面的for循环去掉
```python
Z=np.dot(w.T,X)+b
A=sigma(Z)
dZ=A-Y
dW=np.dot(X,dZ.T)/m
db=np.sum(dZ)/m

w=w-alpha*dW
b=b-alpha*db
```
就这样，越来越简单了不是吗？
上面代码中大写字母代表的就矩阵。

好啦，就是这样，Logistic回归是不是很简单呢？
真正的python代码请见此地址下的其他代码文件。

拜拜！