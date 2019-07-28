# 1、sklearn 数据集与估计器

### sklearn数据集

1、数据集划分

​	训练集		、	测试集

​		7			|			3

​		8			|			2

​		75  		|		   25

建立模型	   |		评估模型



数据集划分API：

+ sklearn.model_selection.train_test_split



2、sklearn数据集接口介绍

+ sklearn.datasets
  + 加载获取流行数据集
  + datasets.load_*()
    + 获取小规模数据集，数据包含在datasets里
  + datasets.fetch_*(data_home=None)
    + 获取大规模数据集，需要从网络上下载，函数的第一个参数是data_home，表示数据集下载的目录，默认是~/scikit_learn_data/





获取数据集返回的类型

+ load* 和 fetch* 返回的数据类型都是 datasets.base.Bunch（字典格式）
  + data：特征数据数组，是 [n_samples * n_features] 的二维 numpy.ndarray 数组
  + target：标签数组，是 n_samples 的一维 numpy.ndarray 数组
  + DESCR：数据描述
  + feature_names：特征名，新闻数据，手写数字、回归数据集没有
  + target_names：标签名



3、sklearn分类数据集

sklearn.datasets.load_iris() 加载并返回鸢尾花数据集

sklearn.datasets.load_digits() 加载并返回数字数据集



+ sklearn.model_selection.train_test_split(*arrays, **options)
  + x    数据集的特征值
  + y    数据集的标签值
  + test_size    测试集的大小，一般为 float
  + random_state    随机数种子，不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
  + return    训练集特征值，测试集特征值，训练标签，测试标签（默认随机取）





用于分类的大数据集：

+ sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train')
  + subset：'train' 或者 'test'，'all'，可选。选择要加载的数据集。训练集的“训练”，测试集的“测试”，两者的“全部”。
+ datasets.clear_data_home(data_home=None)
  + 清除目录下的数据











4、sklearn回归数据集







# 2、分类算法 -- k近邻算法





# 3、k-近邻算法实例





# 4、分类模型的评估





# 5、分类算法 -- 朴素贝叶斯算法







# 6、朴素贝叶斯算法实例







# 7、模型的选择与调优







# 8、决策树与随机森林





