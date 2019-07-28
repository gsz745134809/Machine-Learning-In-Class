# 机器学习

## 01-机器学习介绍

## 02-特征工程和文本特征提取

### 数据集的组成

机器学习的数据：文件 如 csv

mysql：1、性能瓶颈，读取速度；2、格式不太符合机器学习要求数据的格式



pandas：读取工具，基于 numpy：释放了GIL



### 数据集的结构

#### 可用数据集

scikit-learn

Kaggle

UCI

常用数据集数据的结构组成

：结构： 特征值 + 目标值



pandas：一个数据读取非常方便以及基本的处理格式的工具

sklearn：对于<font color=red>特征的处理</font>提供了强大的接口



特征工程：特征工程是将原始数据转换为更好地代表预测模型的潜在问题的特征的过程，从而提高了对未知数据的预测准确性。





数据的特征抽取、数据的特征预处理、数据的降维



### 字典特征抽取：

作用：对字典数据进行特征值化

类：sklearn.feature_extraction.DictVectorizer



sparse矩阵：只写出有值位置的坐标和值，省略为0的地方。节约内存，方便读取处理。



字典数据抽取：把字典中一些类别数据，分别进行转换成特征



### 文本特征抽取

作用：对文本数据进行特征值化

类：sklearn.feature_extraction.text.CountVectorizer

（以空格分开）

对于单个英文字母不统计。



处理中文，采用jieba分词

使用：

```py
import jieba
jieba.cut("我是一个好程序员")
```

返回值：词语生成器。



### tf   *   idf  : 重要性程度

tf：term frequency：词的频率

idf：inverse document frequency：逆文档频率  log(总文档数量/该词出现的文档数量)



TF-IDF：

TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

TF-IDF作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的<font color=red>重要程度</font>。

类：sklearn.feature_extraction.text.TfidfVectorizer

## 03-数据特征预处理

对数据进行处理

特征处理：通过特定的统计方法（数学方法）将数据转换成算法要求的数据



数值型数据：标准缩放：<font color=red>1、归一化</font>；<font color=red>2、标准化</font>；3、缺失值；

类别型数据：one-hot编码

时间类型：时间的切分





sklearn特征处理API：

+ sklearn.preprocessing



#### 归一化：

特点：通过对原始数据进行变换把数据映射到（默认为[0,1]）之间。

#### 公式：

$$
X'\ =\ \frac{x-min}{max-min}\\
X''\ =\ X'*(mx-mi)+mi
$$
注：作用于每一列，max为一列的最大值，min为一列的最小值，那么X'' 为最终结果，mx，mi分别为指定区间值，默认mx为1，mi为0。



sklearn归一化API：

+ sklearn归一化API：sklearn.preprocessing.MinMaxScaler



特征同等重要的时候：进行归一化

目的：使得某一个特征对最终结果不会造成更大影响



问题：如果数据中异常点较多，会有什么影响？

异常点对最大值最小值影响太大



因此，归一化对异常点处理的不好



#### 归一化总结

注意在特定场景下最大值最小值是变化的，另外，最大值与最小值非常容易受<font color=red>异常点</font>影响，所以这种方法鲁棒性较差，只适合<font color=red>传统精确小数据场景</font>。



鲁棒性：反应产品稳定性



#### 标准化

<font color=red>1、特点：通过对原始数据进行变换把数据变换到均值为 0，标准差为 1 范围内</font>。

2、公式：
$$
X'\ =\ \frac{x-mean}{\sigma}
$$
注：作用于每一列，mean 为平均值，$\sigma$ 为标准差

var称为方差，$var=\frac{(x_1-mean)^2+(x_2-mean)^2+...}{n(每个特征的样本数)}，\sigma=\sqrt{var}$

其中：方差（<font color=red>考量数据的稳定性</font>）





结合归一化来谈标准化

对于归一化来说：如果出现异常点，影响了<font color=red>最大值和最小值</font>，那么结果显然会发生改变。

对于标准化来说：如果出现异常点，由于<font color=red>具有一定数据量，少量的异常点对于平均值的影响并不大</font>，从而方差改变较少。



+ sklearn特征化API：scikit-learn.preprocessing.StandardScaler



#### 标准化总结

在已有<font color=red>样本足够多的情况下比较稳定</font>，适合现代嘈杂大数据场景。





#### 缺失值

如何处理数据中的缺失值？

1、删除 ；  2、填补 ；

| 删除                                                         | <font color=red>插补</font>                        |
| ------------------------------------------------------------ | -------------------------------------------------- |
| 如果每列或者行数据缺失值达到一定的比例，建议放弃整行或者整列 | 可以通过缺失值每行或者每列的平均值、中位数来填充。 |



sklearn缺失值API：sklearn.preprocessing.Imputer



Imputer语法

+ Imputer(missing_values='NaN', strategy='mean', axis=0)
  + 完成缺失值插补



关于 np.nan(np.NaN)

1、numpy 的数组中可以使用 np.nan/np.NaN 来代替缺失值，<font color=red>属于 float 类型</font>

2、如果是文件中的一些缺失值，可以替换成 nan，通过 np.array 转化成 float 型的数组即可。







## 04-数据降维

维度：特征的数量

数据降维：减少特征的数量



### 1、特征选择



特征选择的原因：

+ 冗余：部分特征的相关度高，容易消耗计算性能
+ 噪声：部分特征对预测结果有影响



#### 主要方法（三大武器）：

+ <font color=red>Filter（过滤式）：VarianceThreshold</font>
+ <font color=red>Embedded（嵌入式）：正则化、决策树</font>
+ Wrapper（包裹式）



(variance:方差，过滤式：对方差过滤)

方差小代表数据差别不大



sklearn特征选择API：

+ sklearn.feature_selection.VarianceThreshold

VarianceThreshold(threshold=0.0)

+ 删除所有低方差特征





### 2、主成分分析



sklearn主成分分析API：

+ sklearn.<font color=red>decomposition</font>



#### PCA是什么：

本质：PCA是一种分析、简化数据集的技术

目的：是数据维数压缩，尽可能降低原数据的维数（复杂度），<font color=red>损失少量信息</font>。

<font color=red>作用：可以削减回归分析或者聚类分析中特征的数量</font>





PCA：特征数量达到上百的时候--》考虑数据的简化

数据也会改变





高维度数据容易出现的问题：

+ 特征之间通常是<font color=red>相关的</font>





#### PCA语法：

+ PCA(n_components=None)
  + 将数据分解为较低维数空间



