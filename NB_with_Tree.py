import pandas as pd
import numpy as np

import jieba
import jieba.analyse

from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from MyUtils import load_stop_words, predict_top_k, accuracy

if __name__ == '__main__':
    # 导入数据：
    raw_data = pd.read_csv(
        '../Data/CloudSOP_Data/cloudsop_2020_data_tidied.csv')
    raw_data.drop('Unnamed: 0', axis=1, inplace=True)

    # 导入stopwords:
    stopwords = load_stop_words('hit_stopwords.txt')

    # 清理数据：
    print('Tidying the data....\n')
    df = raw_data.dropna(axis=0, subset=['MODULE'])  # 移除MODULE为空的数据
    df['RVERSION'] = df.RVERSION.fillna('Empty')  # 填充R版本为空数据
    df.drop_duplicates(subset=['NO', 'MODULE', 'RVERSION'],
                       inplace=True)  # 移除重复问题单
    df['MODULE'] = df.MODULE.apply(int)
    df['MODULE'] = df.MODULE.apply(str)

    # 将简要描述向量化：
    print('Vectorizing brief description...\n')
    vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stopwords)
    vectorizer.fit(df.DISCRIPT)
    vectored_desc = vectorizer.transform(df.DISCRIPT)

    # 将R版本进行onehot编码：
    print('Encoding R version...\n')
    enc = OneHotEncoder()
    encoded_r = enc.fit_transform(df.RVERSION.to_numpy().reshape(-1, 1))

    # train_test_split:
    kf = KFold(n_splits=10, random_state=0)

    # 开始训练并得到准确率：
    print('Start K-Fold')
    fold = 0
    accs_with_tree = []
    accs_without_tree = []
    for train_index, test_index in kf.split(df.MODULE):
        desc_train = vectored_desc[train_index]
        r_train = encoded_r[train_index]
        module_train = df.MODULE.to_numpy()[train_index].reshape(-1, 1)

        desc_test = vectored_desc[test_index]
        r_test = encoded_r[test_index]
        module_test = df.MODULE.to_numpy()[test_index].reshape(-1, 1)

        # 训练NB DT：
        cnb = ComplementNB()
        cnb.fit(desc_train, module_train)

        dt = DecisionTreeClassifier()
        dt.fit(r_train, module_train)

        # 计算概率：
        desc_proba = cnb.predict_proba(desc_test)
        r_proba = dt.predict_proba(r_test)

        # 为了避免模块问题单出现频率对推荐结果造成影响，转换为0-1的向量，只做物理隔离：
        r_proba[r_proba > 0.000001] = 1
        joint_proba = desc_proba * r_proba

        # 推荐top3：
        item_tree = predict_top_k(joint_proba, cnb.classes_)
        item_none_tree = predict_top_k(desc_proba, cnb.classes_)

        # 计算accuracy：
        acc1 = accuracy(module_test, item_tree)
        acc2 = accuracy(module_test, item_none_tree)
        accs_with_tree.append(acc1)
        accs_without_tree.append(acc2)
        print('========FOLD {}========'.format(fold+1))
        print('Accuracy with Decision Tree:\t{}'.format(acc1))
        print('Accuracy without Decision Tree:\t{}\n'.format(acc2))
        fold += 1
