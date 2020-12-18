import re
import heapq
import numpy as np
from suds.client import Client


def load_stop_words(file):
    stopwords = []
    with open(file, encoding='utf-8') as f:
        while True:
            word = f.readline()
            word = re.sub(r'\s', '', word)
            # print(word)
            if word:
                stopwords.append(word)
            elif not word:
                break
    return stopwords


def get_module_name(module_number):
    pbi_systemId = 'CMIR_PBI_Common,,,A1CzEBjF0E'
    pbi_url = 'http://nkweb-sit.huawei.com/pbiws/cxfservices/pbiWebservice?wsdl'
    client = Client(pbi_url)
    moduleInfo = client.service.queryPBIEntitiesNew(
        {
            'category': '434-00024645',
            'id': module_number
        }, pbi_systemId)
    return moduleInfo


def predict_top_k(proba, modules, k=3):
    res = np.zeros(shape=(proba.shape[0], k), dtype=np.object)

    for i, p in enumerate(proba):
        n_largest_proba = heapq.nlargest(k, p)
        # 处理概率为零的情况：
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


def get_module_name(module_number):
    pass
