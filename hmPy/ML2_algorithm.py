from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups


li = load_iris()

# print("获取特征值")
# print(li.data)
# print("目标值")
# print(li.target)
# print("描述")
# print(li.DESCR)

# # 注意返回值，
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
# print("训练集特征值和目标值", x_train, y_train)
# print("测试集特征值和目标值", x_test, y_test)


news = fetch_20newsgroups(subset='all')

print(news.data)
print(news.target)

