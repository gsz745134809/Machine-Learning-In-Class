# 分类算法 - k近邻算法（KNN）
通过你的“邻居”来判断你的类别

### 定义：

如果一个样本在特征空间中的<font color=red>k个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别</font>，则该样本也属于这个类别。

### 来源：
KNN 算法最早是由 Cover 和 Hart 提出的一种分类算法。


### 计算距离公式：
两个样本的距离可以通过如下公式计算，又叫<font color=red>欧氏距离</font>，
比如说，
$$
a(a1,a2,a3), b(b1,b2,b3) \\
\sqrt{(a1-b1)^2+(a2-b2)^2+(a3-b3)^2}
$$

### K-近邻算法需要做标准化处理

### sklearn  k-近邻算法API

+ sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm=‘auto’)
  + n_neighbors：int，可选（默认=5），k_neighbors 查询默认使用的邻居数
  + algorithm：{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}，可选用于计算最近邻的算法：‘ball_tree’将会使用 BallTree，‘kd_tree’将使用KDTree。‘auto’将尝试根据传递给 fit 方法的值来决定最合适的算法。（不同实现方式影响效率）





K-近邻算法优缺点



+ 优点：
  + 简单，易于理解，易于实现，<font color=red>无需估计参数，无需训练</font>
+ 缺点：
  + 懒惰算法，对测试样本分类时的计算量大，内存开销大
  + 必须指定 K 值，K 值选择不当则分类精度不能保证。
+ 使用场景：
  + 小数据场景，几千～几万样本，具体场景具体业务去测试。





加快搜索速度 ---- 基于算法的改进 KDTree，API接口里面有实现



















