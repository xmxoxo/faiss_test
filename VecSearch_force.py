#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

# 向量搜索 暴力算法


import numpy as np
import time
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

# 把字向量转化为句向量，简单相加
def seg_vector (txt, dict_vector, emb_size=768):
    seg_v = np.zeros(emb_size)
    for w in txt:
        if w in dict_vector.keys():
            v = dict_vector[w]
            seg_v += v
    return seg_v


# 余弦相似度各种算法： CosSim_dot最快
def CosSim(a, b):
    return 1-cosine(a, b)

def CosSim_sk(a,b):
    score = cosine_similarity([a,b])[0,1]
    return score

CosSim_dot = lambda a,b : np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def CosSim_np (a, b):
    a = np.mat(a)
    b = np.mat(b)
    num = float(a.T * b) #若为行向量则 A * B.T
    #num = float(a * b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = num / denom #余弦值
    #sim = 0.5 + 0.5 * cos 
    sim = 1 - cos 
    return sim



'''
def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 0))#.to(device)
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 0))#.to(device)
    pooled_mul_12 = tf.reduce_sum(q * a, 0)#.to(device)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")#.to(device)
    with tf.Session() as sess:
        cos = sess.run(score)#.to(device)
    return cos

# 在TF上计算余弦相似度
def get_cos_distance(X1, X2):
    # calculate cos distance between two sets
    # more similar more big
    (k,) = X1.shape
    (m,) = X2.shape
    # 求模
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
    # 内积
    X1_X2 = tf.matmul(X1, tf.transpose(X2))
    X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[k,1]),tf.reshape(X2_norm,[1,m]))
    # 计算余弦距离
    cos = X1_X2/X1_X2_norm
    return cos
'''

# 向量搜索类
class VecSearch:
    def __init__(self):
        self.dicts = {}

    # 返回当前总共有多少个值
    def curr_items ():
        return len(self.dicts)
    
    # 添加文档
    def add_doc (self, key, vector):
        self.dicts[key] = vector

    # 查找向量, 
    # 返回结果为 距离[D], 索引[I]
    def search(self, query, top=5):
        # 返回结果，结构为：[sim, key]
        ret = np.zeros((top,2))
        # 计算余弦相似度最大值
        for key, value in self.dicts.items():
            sim = CosSim_dot(query, value)
            #sim = CosSim(query, value)
            #sim = CosSim_sk(query, value)
            #sim = cosine(query, value)
            #print(sim)
            if sim > ret[top-1][0]:
                b = np.array([[sim, key]]).astype('float32')
                ret = np.insert(ret, 0, values=b, axis=0)
                # 重新排序后截取 
                idex = np.lexsort([-1*ret[:,0]])
                ret = ret[idex, :]
                ret = ret[:top,]
                #print(ret)
                #print('-'*40)
        return ret[:,0], ret[:,1].astype('int')

#-----------------------------------------
# 测试
def test ():
    np.random.seed(1234)             # make reproducible
    print('大批量向量余弦相似度计算-[暴力版]'.center(40,'='), flush=True)
    # 随机生成10万个向量
    total = 100000
    dim = 768
    print('随机生成%d个向量，维度：%d' % (total, dim), flush=True)
    #rng = np.random.RandomState(0)
    #X = rng.random_sample((total, dim))  
    X = np.random.random((total, dim))
    X[:, 0] += np.arange(total) / 1000.

    #print('前10个向量为：')
    #print(X[:10])
    print('正在创建搜索器...')
    start = time.time()

    # 创建搜索器
    vs = VecSearch()
    # 把向量添加到搜索器
    for i in range(total):
        vs.add_doc(i, X[i])
    end = time.time()
    total_time = end - start
    print('添加用时:%4f秒' % total_time)

    # 查看当前进程使用的内存情况
    import os,psutil

    process = psutil.Process(os.getpid())
    print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
    
    # 进行测试
    print('单条查询测试'.center(40,'-'))
    test_times = 100
    #Q = rng.random_sample((test_times, dim))
    Q = np.random.random((test_times, dim))
    Q[:, 0] += np.arange(test_times) / 1000.

    q = Q[0]
    D, I = vs.search(q)
    #print('索引号:%d, 余弦相似度:%f' % r)
    print('搜索结果:', D, I)

    # 显示详细结果
    def showdetail (X,q,D,I):
        print('显示查询结果，并验证余弦相似度...')
        for i, v in enumerate(I):
            #np.squeeze(X[v])
            #c = CosSim_dot(Q[0], X[v])
            r = (v, D[i]) # CosSim_dot(Q[0], X[v]), #
            print('索引号:%5d, 距离:%f' % r ) #, 余弦相似度:%f
            #rv = X[v][:10]
            #print('\n查询结果(超长只显示前10维:%s' % rv)

    showdetail (X,q,D,I)

    print('批量查询测试'.center(40,'-'))
    start = time.time()
    print('批量测试次数：%d 次，请稍候...' % test_times )
    for i in range(test_times):
        r = vs.search(Q[i])

    end = time.time()
    #print((end-start), (end-start)/test_times)
    total_time = end - start
    print('总用时:%d 秒, 平均用时:%4f 毫秒' % (total_time, total_time*1000/test_times) )
    return

    # 人工测试
    while 1:
        print('-'*40)
        txt = input("回车开始测试(Q退出)：").strip()
        if txt.upper()=='Q': break
      
        # 随机生成一个向量
        print('随机生成一个查询向量...')
        q = rng.random_sample(dim)
        print("query:%s..." % q[:10])

        # 查询
        start = time.time()
        r = vs.search(q)
        print('查询结果:')
        print('索引号:%d,相似度:%f' % r) # , X[r]
        end = time.time()
        total_time = end - start
        print('总用时:%d 秒, 平均用时:%4f 毫秒' % (total_time, total_time*1000) )

if __name__ == '__main__':
    pass
    test()

