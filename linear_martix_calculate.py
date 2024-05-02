import torch
#线性代数预备知识

#矩阵的引入以及矩阵的转置
'''
#A = th.arange(20).reshape(5,4) #生成一个5乘4的张量，元素按0-19
#print(A,'\n',A.T) #打印A 和 A的转置
#B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
#print(B == B.T)
'''

#两个矩阵按元素相乘
'''A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
#hadamard product 按元素相乘
#print(A*B)'''

#点积
'''
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y),torch.sum(x*y)) #按元素相乘在相加
'''

#矩阵乘法
'''
#矩阵乘向量
x = torch.arange(4, dtype=torch.float32)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.shape, x.shape, torch.mv(A, x)) #用矩阵乘向量用torch.mv(M,V)
#矩阵乘矩阵
B = torch.ones(4, 3)
print(A,'\n',B,'\n',torch.mm(A, B))
'''

#向量范数和矩阵范数
'''
在线性代数中，向量范数是将向量映射到标量的函数$f$。
给定任意向量$\mathbf{x}$，向量范数要满足一些属性。
第一个性质是：如果我们按常数因子$\alpha$缩放向量的所有元素，
其范数也会按相同常数因子的*绝对值*缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二个性质是熟悉的三角不等式:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三个性质简单地说范数必须是非负的:

$$f(\mathbf{x}) \geq 0.$$
'''

'''
#矩阵向量范数
#向量范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))#L2范数，that is 所有元素平方求和后开根号
print(torch.abs(u).sum())#L1 所有元素的绝对值求和

#矩阵范数
#常用的是F范数，原理是将矩阵拉成向量后对向量的每个元素进行平方求和处理，最后在对求和项开根号。
print(torch.ones((4, 9)))
print(torch.norm(torch.ones((4, 9))))
'''

#练习题目

#1.证明矩阵(A^T)^T = A
#证明思路可以是预设一个矩阵的形式，然后通过转置运算的定义来证明。
#2.类似于题1
#3.总是对称的，证明思路就是（A+A^T）^T = A+A^T

'''
#4.本节中定义了形状$(2,3,4)$的张量`X`。`len(X)`的输出结果是什么？
X = torch.arange(24).reshape(2,3,4)
print(len(X))#我认为输出结果就是24，因为元素总个数就是24个
#结果是2！！！
#我明白了，len会返回的只是维度！也就是第一个维度2
#如果想要获取到所有元素的个数，请用如下指令
print(X.numel()) #numel 返回的是张量中所有元素的个数

#5.len返回的是第一个维度的个数
#6.运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因？
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.sum(axis=1,keepdim = True)) #按照默认的广播机制，这个是无法广播的；
'''
'''
#考虑一个具有形状$(2,3,4)$的张量，在轴0、1、2上的求和输出是什么形状?
X = torch.arange(24).reshape(2,3,4)
print(X.sum(axis=2))
'''

'''
#7.为`linalg.norm`函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么?
X = torch.tensor([[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]],[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]]])
print(X,X.shape)
print(torch.norm(X))
#F范数
'''