### tf-idf + Naiive Bayes + Decision Tree实现根据DTS单简要描述定位到模块

## 0. 背景描述

一个小课题。现在能拿到几个产品的DTS问题单，以及iSource上面提交的信息。在提单时一般会需要选择问题对应的子系统、特性、模块，现在希望做到根据提单时的简要描述，推荐出可能出现问题的模块、特性、子系统。

## 1. 文本tf-idf向量化

使用sklearn中的 **TfidfVectorizer** 训练。为了实现更好的中文分词，tokenizer选择jieba.cut：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

```python
vectorizer = TfidfVectorizer(tokenizer = jieba.cut, stop_words = stop_words)
vectorizer.fit(df.DISCRIPT)
```



stop_words指定为本地的stop_words文件

文本分词后结果：

![df_head](D:\学习资料\NLP\文本向量化\pics\df_head.PNG)

然后使用训练好的vectorizer生成一个稀疏矩阵， 并划分训练集和测试集：

```python
from sklearn.model_selection import train_test_split
```

```python
vectored_desc = vectorizer.transform(df.DISCRIPT)
x_train, x_test, y_train, y_test = train_test_split(vectored_desc, df.MODULE,test_size = 0.2, random_state = 128)
```



## 2. Naiive Bayes：

NB最基本的假设是各词语的出现是相互独立的，这也是被称之为朴素的原因。不同的NB分类器区别在于处理:

$P(x_i|y)$时的假设。

最后选择了两种Bayes模型： multinomial和complement，两种都是针对文本类训练较好的模型，其中complement是MNB的改进，适用于不平衡的数据集。

### 2.1 模型训练

```python
from sklearn.naive_bayes import ComplementNB,  MultinomialNB
from sklearn.model_selection import cross_val_score
```

```python
# 初始化、训练
CNB = ComplementNB()
MNB = MultinomialNB()
CNB.fit(x_train, y_train)
MNB.fit(x_train, y_train)

# 预测
y_predict1 = CNB.predict(x_test)
y_predict2 = MNB.predict(x_test)
```

###2 .2 模型评价

```python
# 打分（accuracy）
CNB.score(x_test, y_test), MNB.score(x_test, y_test) # 0.749178320551653, 0.6284075530063801

# 交叉验证：
scores1 = cross_val_score(CNB, sparse_matrix, df.MODULE, cv = 8)
scores2 = cross_val_score(MNB, sparse_matrix, df.MODULE, cv = 8)

'''
(array([0.53701794, 0.56434316, 0.51649825, 0.52742834, 0.49381316,
        0.51603589, 0.50500155, 0.46158606]),
 array([0.4383378 , 0.49907197, 0.46525057, 0.45473293, 0.4265828 ,
        0.41940806, 0.45457358, 0.43405177]))
'''
```



综合来看，CNB的平均accuracy明显高于MNB，但是效果依然不是很理想。

### 2.3 模型验证

在这里，考虑到业务的实际情况：目标是为了推荐可选的匹配的模块，所以在计算准确度时，只需要保证推荐的多个选项能命中目标即可，所以这里我更改了计算准确率的方式，即：根据简要描述，推荐top3结果，只要推荐结果中命中就算推荐成功。

```python
def predict_top_k(proba, modules, k=3):
    res = np.zeros(shape=(proba.shape[0], k), dtype=np.object)

    for i, p in enumerate(proba):
        n_largest_proba = heapq.nlargest(k, p)
        # 处理概率大量为零导致推荐模块大于K的情况：
        n_largest_proba = list(filter(lambda x: x != 0, n_largest_proba))
        
        idx = np.where(np.isin(p, n_largest_proba) == True)
        n_largest_item = modules[idx]
        tmp = n_largest_item

        res[i][:len(tmp)] = tmp
    return res

def accuracy(y_true, y_predict):
    count = 0
    for i in range(y_true.shape[0]):
        if y_true[i] in y_predict[i]:
            count += 1

    accuracy = count / y_true.shape[0]
    return accuracy
```



考虑到在 **2.2** 中结论为CNB效果优于MNB，所以基于CNB使用交叉验证，评估模型。选取fold = 10


```python
from sklearn.model_selection import KFold
def kfold(clf, x, y, k):
    '''
    clf: classifier.
    k: number of folds
    '''
    accs = []

    kf = KFold(n_splits = k)
    i = 1
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(x_train, y_train)
        y_predict = clf.predict_top_k(x_test, k = 3)

        acc = clf.accuracy(y_test, y_predict)
        accs.append(acc)
        print('FOLD: {} ======= Accuracy: {}'.format(i, acc))
        i += 1

    return accs
```

```python
kfold(CNB(), sparse_matrix, df.MODULE.to_numpy(), 10)
```

结果如下：

```python
FOLD: 1 ======= Accuracy: 0.6380976930016755
FOLD: 2 ======= Accuracy: 0.6158803815416345
FOLD: 3 ======= Accuracy: 0.580046403712297
FOLD: 4 ======= Accuracy: 0.41505542665635475
FOLD: 5 ======= Accuracy: 0.4529517916988915
FOLD: 6 ======= Accuracy: 0.44753802526424336
FOLD: 7 ======= Accuracy: 0.3993297241557102
FOLD: 8 ======= Accuracy: 0.4466357308584687
FOLD: 9 ======= Accuracy: 0.674400618716164
FOLD: 10 ======= Accuracy: 0.6607373034287187
```

可以看到受数据划分方式的影响还是较大的，需要继续改进。

实际操作使用一下模型，

- 选择问题单：**DTS2020121400802**

- 模块：**Incident生命周期管理**

- 模块PBI编号：**1110352547** 

- 简要描述：**【AIFM】【DCN】DCN需要支撑新的License部署场景，无平台License服务时，产品自行控制是否能够进入Incident界面**

  ```python
  CNB.predict_top_k(vectorizer.transform(['【AIFM】【DCN】DCN需要支撑新的License部署场景，无平台License服务时，产品自行控制是否能够进入Incident界面']))
  # 文本向量化之后用CNB预测
  # array([['1110352384', '1110352420', '1110352546']], dtype=object)
  ```

  可以看到推荐出的第三个结果命中（PBI中显示的编码和API获取的编码差1）
  
  
  
### 2.4 模型改进

#### 2.4.1 使用决策树概率实现R版本隔离

方案1：通过代码直接实现隔离，保存多个R版本的模型。即：数据进来之后，先判断R版本，然后调出对应R版本的模型，进行预测.

方案2：通过代码实现隔离，只保存全量的模型。数据输入后预测在各模块下的概率，强制将不属于该R版本的模块的概率置0，然后推出topK.

方案3：通过模型实现隔离，引入决策树。先将R版本输出随机树得到该R版本对应各模块的概率，根据随机树的构造原理，可以预见，各R版本下对应各模块的概率实际就是其频率；基于此，再将简要描述输入到NB模型中，得到基于简要描述得到的各模块的概率；两组概率相乘，再重新计算概率即可。

综合考虑，使用方案3.

**清洗数据**

因为引入了新的关键字段，所以要重新清洗数据：

```python
# 清理数据：
print('Tidying the data....\n')
df = raw_data.dropna(axis=0, subset=['MODULE'])  # 移除MODULE为空的数据
df['RVERSION'] = df.RVERSION.fillna('Empty')  # 填充R版本为空数据
df.drop_duplicates(subset=['NO', 'MODULE', 'RVERSION'],inplace=True)  # 移除重复问题单
df['MODULE'] = df.MODULE.apply(int)
df['MODULE'] = df.MODULE.apply(str)
```

**OneHot 编码**

首先，将R版本进行OneHot编码，将文字信息转化成一个sparse matrix——随机树可以接受的输入：

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
encoded_r = enc.fit_transform(df.RVERSION.to_numpy().reshape(-1, 1))
```

**生成新的概率**

将NB的概率结合待预测样本的R版本信息从决策树得到的概率生成新的概率：
$$
P_{joint}(module_i) = \frac{P_{rversion}(i)*P_{desc}(i)}{\sum_{i} P_{rversion}(i)*P_{desc}(i)}
$$
进而根据新的联合概率，推荐出topK

**模型训练与验证**

依旧采用**k-fold**的方法：

```python
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
    # ========================模型训练===========================
    # 训练NB DT：
    cnb = ComplementNB()
    cnb.fit(desc_train, module_train)

    dt = DecisionTreeClassifier()
    dt.fit(r_train, module_train)

    # 计算概率：
    desc_proba = cnb.predict_proba(desc_test)
    r_proba = dt.predict_proba(r_test)
    
    joint_proba = desc_proba * r_proba

    # 推荐top3：
    item_tree = predict_top_k(joint_proba, cnb.classes_)
    item_none_tree = predict_top_k(desc_proba, cnb.classes_)

    # =======================模型验证========================
    # 计算accuracy：
    acc1 = accuracy(module_test, item_tree)
    acc2 = accuracy(module_test, item_none_tree)
    accs_with_tree.append(acc1)
    accs_without_tree.append(acc2)
    print('========FOLD {}========'.format(fold+1))
    print('Accuracy with Decision Tree:\t{}'.format(acc1))
    print('Accuracy without Decision Tree:\t{}\n'.format(acc2))
    fold += 1
```

结果如下：

```python
Start K-Fold
========FOLD 1========
Accuracy with Decision Tree:	0.24608150470219436
Accuracy without Decision Tree:	0.6943573667711599

========FOLD 2========
Accuracy with Decision Tree:	0.25313479623824453
Accuracy without Decision Tree:	0.7045454545454546

========FOLD 3========
Accuracy with Decision Tree:	0.2468652037617555
Accuracy without Decision Tree:	0.6575235109717869

========FOLD 4========
Accuracy with Decision Tree:	0.23981191222570533
Accuracy without Decision Tree:	0.6300940438871473

========FOLD 5========
Accuracy with Decision Tree:	0.2421630094043887
Accuracy without Decision Tree:	0.6559561128526645

========FOLD 6========
Accuracy with Decision Tree:	0.22648902821316613
Accuracy without Decision Tree:	0.5666144200626959

========FOLD 7========
Accuracy with Decision Tree:	0.23686274509803923
Accuracy without Decision Tree:	0.5435294117647059

========FOLD 8========
Accuracy with Decision Tree:	0.23215686274509803
Accuracy without Decision Tree:	0.56

========FOLD 9========
Accuracy with Decision Tree:	0.24392156862745099
Accuracy without Decision Tree:	0.6219607843137255

========FOLD 10========
Accuracy with Decision Tree:	0.1388235294117647
Accuracy without Decision Tree:	0.668235294117647
```

可以看到操作中比起只使用NB模型，准确率大大下降，这是不合理的结果，因为通过R版本隔离天然筛选了一些错误选项，所以准确率最差的情况下也只是不提升。所以仔细评估了一下，还是因为犯了一个小错误：主要原因是因为个别模块出现频率较高，直接使用决策树的概率做乘积，虽然实现了基于R版本的隔离，但是使得模块出现的频率影响了NB分类模型基于简要描述的生成的概率（这个概率是作为推荐相似的基本原则）。因此为了消除这个影响还需要进一步改进，使得决策树仅实现对R版本的隔离，不影响NB模型在该R版本下计算的各模块对应的概率。

#### 2.4.2 使用决策树转化0-1向量实现R版本隔离

基于2.4.1中提到的问题，将决策树输出的概率进行0-1二值处理：概率大于零设置为1，这样决策树在模型中的作用仅为隔离R版本，而不对概率本身产生影响。添加代码

```python
r_proba[r_proba > 0.000001] = 1
```

结果如下：

```pyhon
Start K-Fold
========FOLD 1========
Accuracy with Decision Tree:	0.7210031347962382
Accuracy without Decision Tree:	0.6943573667711599

========FOLD 2========
Accuracy with Decision Tree:	0.7241379310344828
Accuracy without Decision Tree:	0.7045454545454546

========FOLD 3========
Accuracy with Decision Tree:	0.6943573667711599
Accuracy without Decision Tree:	0.6575235109717869

========FOLD 4========
Accuracy with Decision Tree:	0.6590909090909091
Accuracy without Decision Tree:	0.6300940438871473

========FOLD 5========
Accuracy with Decision Tree:	0.6873040752351097
Accuracy without Decision Tree:	0.6559561128526645

========FOLD 6========
Accuracy with Decision Tree:	0.627742946708464
Accuracy without Decision Tree:	0.5666144200626959

========FOLD 7========
Accuracy with Decision Tree:	0.5937254901960785
Accuracy without Decision Tree:	0.5435294117647059

========FOLD 8========
Accuracy with Decision Tree:	0.6235294117647059
Accuracy without Decision Tree:	0.56

========FOLD 9========
Accuracy with Decision Tree:	0.6572549019607843
Accuracy without Decision Tree:	0.6219607843137255

========FOLD 10========
Accuracy with Decision Tree:	0.6792156862745098
Accuracy without Decision Tree:	0.668235294117647
```

可以看到，相较于没有使用决策树来实现R版本隔离的模型，准确度有了进一步的提升。

## 3. 可以改进的方向：

1. 文本向量化本身，虽然tfidf比起one-hot，考虑到了词频与重要性的关系，但是实际上对于各词语之间的关系并没有很好的反应，可以考虑glove、word2vec等向量化方法。
2. 可以考虑针对业务本身生成词典，当前使用的词典还是jieba中默认的词典，以及tf-idf值，所以可能对于业务具体场景，还有优化的空间。
3. 词库选取。

## 4. 总结

至此，我的一个基于 Naiive Bayes 和 tf-idf 的文本分类的简单模型就算完成了。最开始选择NB这个模型的原因很简单：当前比较成熟的垃圾邮件分类算法的基础都是基于NB实现的，在这个问题中，我寻思着本质也是基于文本分类，只不过把垃圾邮件分类的二分类问题变成了多分类，再加上朴素贝叶斯这个方法，本身实现简单，可解释性强，所以想优先使用该方法实现一下。在这个过程中还是参考了一些方法，当然最主要的还是基于sklearn编程，不过这个过程中感觉重点在于对于不同模型的理解与组合应用，而不在于模型是不是自己代码实现的。下一步目标是采用其他的、“新潮的”向量化方法，比较一下和Naiive Bayes这种传统机器学习方法之间效果的差别。








