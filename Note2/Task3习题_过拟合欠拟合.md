# 线性回归

![1582245661400](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582245661400.png)

**解答**：参考书中定义的线性回归网络函数linreg：

def linreg(X, w, b):

​	return torch.mm(X, w) + b

这里X为(1000,2)，w为(2,1)，b为(1,)。

根据广播机制的原理，在初始化定义b时，不需要指定size，将b定义为(1,)的向量，故选C。D选项也对，但是7是批量的大小，模型的参数形状不应该和批量大小挂钩。

![1582246570822](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582246570822.png)

**解答**：这个问题主要考察广播机制，view函数，以及对损失函数的理解。

1. 首先解释下view函数：view函数的作用是改变原来tensor的张量，这里值得注意的有两个->

​       view(-1)是将tensor平铺，如原tensor是3*4的，平铺后为(12,)；[是(12,)而不是(12,1)]

​	   view(-1,-1)是将tensor转置，但是如果tensor为(n,)时，转置后为(n,1)；

2. 再解释一下原损失函数为什么要这样定义：y_hat为通过Xw+b得到的(10,1)维的向量，y为定义的(10,)的向量。(注意这里是(10,)，区别于(10,1))。那么对于损失函数，为什么不可以直接使用$(y_{hat}-y)^2/2$来进行预测呢？这样的问题在于，通过广播机制，y_hat - y会得到一个10*10的矩阵，而我们的目标是(10,1)的向量，故使用view将y变为(10,1)，再进行相减：y_hat - y.view(y_hat.size())；

3. 广播机制，这里从两个方面来解释下广播机制：一是触发广播机制的条件；二是广播机制的返回值；

   广播机制可以完成不同shape的tensor之间的运算，其需要满足两个条件：

   1. 两个tensor的维数大于等于1；
   2. 从后往前遍历两个tensor的维数，对应位置上的数必须要么相同，要么有一个为1，要么有一个不存在。

   ![1582248141582](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582248141582.png)

   广播机制的返回值，有两个步骤：

   1. 如果两个tensor的维数不相同，则再维数少的tensor前补1；
   2. 接着依次比较同一维度下的值，选择最大的作为返回的维度；

   ![1582248155597](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582248155597.png)

   ![1582248174989](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582248174989.png)

   # Softmax回归

   ![1582248372431](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582248372431.png)

**解答**：这个问题主要考察的是Softmax的运算。

softmax运算是为了表达出样本预测各个输出的概率，得到的矩阵每行元素和为1且非负，最终使得输出矩阵中的任意一行元素表示该样本在各个输出类别上的预测概率，运算步骤包括：

一. 先通过exp函数对每个元素做指数运算;

二. 再对exp矩阵的同行元素求和；

三. 最后令矩阵每行各元素与该行元素之和相除。

那么对于这道题中，就是比较[a,b,c]中，$\frac{e^a}{e^a+e^b+e^c}$的概率给出的是否相同即可。

![1582248700666](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582248700666.png)

**解答**：参考原书中的代码，train的accuracy在一个epoch中完成，test的accuracy在epoch后完成。对于A选项，过拟合的结果应该是train_data的acc高于test_data，而不是低于，故选C。

# 多层感知机

![1582249772097](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582249772097.png)

**解答**：

A选项：softmax回归和线性回归神经网络都是单层，考虑到多层神经网络，在输入层和输出层之间增加隐藏层。但是隐藏层的式子合并后，和之前单层的效果一样，所以才考虑激活函数对其进行非线性变换；

B选项：tanh和sigmoid的函数表达式不一样，详情可以参考Task总结笔记；

C选项：梯度消失还没看到，这个之后补充；

![1582250182417](C:\Users\MAIBENBEN\AppData\Roaming\Typora\typora-user-images\1582250182417.png)

**解答**：题目比较简单，隐藏层的W为(256*256,1000),输出层的W为(1000,10)，加起来后为如上结果。