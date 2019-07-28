from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np


def dictvec():
    """
    字典数据抽取
    """
    # 实例化
    dict = DictVectorizer()

    # 调用 fit_transform
    data = dict.fit_transform([
        {'city': '北京', 'temperature': 25}, 
        {'city': '上海', 'temperature': 30}, 
        {'city': '广州', 'temperature': 35}
    ])

    print(data)

    print(dict.get_feature_names())

    print(dict.inverse_transform(data))

    print(data)
    
    return None


def countvec():
    """
    对文本进行特征值化
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["life is short,i like python", "life is too long,i dislike python"])

    print(data)

    print(cv.get_feature_names())
# ['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']

    print(data.toarray())
# [[0 1 1 1 0 1 1 0]
#  [1 1 1 0 1 1 0 1]]
    return None

# 1、统计所有文章当中所有的词，重复的只看做一次。
# 2、对每篇文章，在词的列表里面进行统计每个词出现的次数。
# （单个字母不统计）


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝大部分是死在明天晚上，所以每个人都不要放弃今天")

    con2 = jieba.cut("我们看到的从很远星系传来的光是在几百万年前发出的，这样当我们看到宇宙时，我们是在看它的过去")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正的秘密取决于如何将其与我们所了解的事物相联系")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)


    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])

    print(data)

    print(cv.get_feature_names())

    print(data.toarray())


def tfidfvec():
    """
    中文特征值化
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(data)

    print(tf.get_feature_names())

    print(data.toarray())


def mm():
    """
    归一化处理
    """
    mm = MinMaxScaler()

    mm1 = MinMaxScaler(range(2,3))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)
# [[1.         0.         0.         0.        ]
#  [0.         1.         1.         0.83333333]
#  [0.5        0.5        0.6        1.        ]]


def stand():
    """
    标准化缩放
    """
    std = StandardScaler()

    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)
# [[-1.06904497 -1.35873244  0.98058068]
#  [-0.26726124  0.33968311  0.39223227]
#  [ 1.33630621  1.01904933 -1.37281295]]


def im():
    """
    缺失值处理
    """
    # NaN, nan
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)
# [[1. 2.]
#  [4. 3.]
#  [7. 6.]]
    

def var():
    """
    特征选择 - 删除低方差的特征
    """
    var = VarianceThreshold(threshold=0.0)  # 删除 方差为 0.0 的特征

    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)
# [[1. 2.]
#  [4. 3.]
#  [7. 6.]]


def pca():
    """
    主成分分析进行特征降维
    """
    pca = PCA(n_components=0.9)  # 保留 90% 特征

    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])

    print(data)
# [[ 1.22879107e-15  3.82970843e+00]
#  [ 5.74456265e+00 -1.91485422e+00]
#  [-5.74456265e+00 -1.91485422e+00]]

if __name__ == "__main__":
    # dictvec()
    # countvec()

    # hanzivec()

    # tfidfvec()

    # mm()

    # stand()

    # im()

    pca()





