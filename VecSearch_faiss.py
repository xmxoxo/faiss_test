#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

# VecSearch
# pip install datasketch
# document, key:integer, vector:np

import argparse
import numpy as np
import time
from scipy.spatial.distance import cosine
#from sklearn.metrics.pairwise import cosine_similarity
import faiss 

np.random.seed(1234)             # make reproducible


# 把字向量转化为句向量，简单相加
def seg_vector (txt, dict_vector, emb_size=768):
    seg_v = np.zeros(emb_size)
    for w in txt:
        if w in dict_vector.keys():
            v = dict_vector[w]
            seg_v += v
    return seg_v


# 余弦相似度
def CosSim(a, b):
    return 1-cosine(a, b)

'''
def CosSim_sk(a,b):
    score = cosine_similarity([a,b])[0,1]
    return score
'''

def CosSim_dot(a, b):
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return score

 
class VecSearch:
    def __init__(self, dim=10, nlist=100, gpu=-1):
        self.dim = dim
        self.nlist = nlist                      #聚类中心的个数
        #self.index = faiss.IndexFlatL2(dim)    # build the index
        quantizer = faiss.IndexFlatL2(dim)      # the other index

        # faiss.METRIC_L2: faiss定义了两种衡量相似度的方法(metrics)，
        # 分别为faiss.METRIC_L2 欧式距离、 faiss.METRIC_INNER_PRODUCT 向量内积
        # here we specify METRIC_L2, by default it performs inner-product search
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        
        try:
            if gpu>=0:
                if gpu==0:
                    # use a single GPU
                    res = faiss.StandardGpuResources()  
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                else:
                    gpu_index = faiss.index_cpu_to_all_gpus(self.index)
            
                self.index = gpu_index
        except :
            pass
        
        # data 
        self.xb = None
    
    # 返回当前总共有多少个值
    def curr_items ():
        # self.index.ntotal
        return self.xb.shape[0]

    # 清空数据
    def reset (self):
        pass
        self.xb = None
        

    # 添加向量，可批量添加，编号是按添加的顺序；
    # 参数: vector, 大小是(N, dim)
    # 返回结果：索引号区间, 例如 (0,8), (20,100)
    def add (self, vector):
        if not vector.dtype == 'float32':
            vector = vector.astype('float32')
        
        if self.xb is None:
            prepos = 0
            # vector = vector[np.newaxis, :]   
            self.xb = vector.copy()
        else:
            prepos = self.xb.shape[0]
            self.xb = np.vstack((self.xb,q))
        
        return (prepos, self.xb.shape[0]-1)

    # 添加后开始训练
    def reindex(self):
        self.index.train(self.xb)
        self.index.add(self.xb)                  # add may be a bit slower as well
    

    # 查找向量, 可以批量查找，
    # 参数：query (N,dim)
    # 返回： 距离D,索引号I  两个矩阵
    def search(self, query, top=5, nprobe=1):
        # 查找聚类中心的个数，默认为1个。
        self.index.nprobe = nprobe #self.nlist 

        # 如果是单条查询，把向量处理成二维 
        #print(query.shape)
        if len(query.shape)==1:
            query = query[np.newaxis, :]
        #print(query.shape)
        # 查询
        if not query.dtype == 'float32':
            query = query.astype('float32')
        D, I = self.index.search(query, top)     # actual search
        return D, I

#-----------------------------------------
# 测试  total=100000, dim=768, test_times=1000, top_n=5 
def test_task (args):
    total = args.total
    dim = args.dim
    test_times = args.test_times
    top_n = args.top_n
    gpu = args.gpu

    print('大批量向量余弦相似度计算-[faiss版]'.center(40,'='))
    # 随机生成向量 1百万   # total = 1000000
    # 向量的维度   # dim = 1024 #768
    print('随机生成%d个向量，维度：%d' % (total, dim), flush=True)
    #rng = np.random.RandomState(0)
    #X = rng.random_sample((total, dim))
    X = np.random.random((total, dim))
    # 把第1列的数变成线性分布；
    X[:, 0] += np.arange(total) / 1000.
    # 保存数据，看下有多大 2020/5/22
    # np.save('dat.npy',X)

    #print('前10个向量为：')
    #print(X[:10])
    print('正在创建搜索器...')
    # 输出GPU使用情况 2020/9/22
    if gpu<0:
        gpuinfo = '不使用'
    elif gpu==0:
        gpuinfo = '单个'
    else:
        gpuinfo = '全部'        
    print('GPU使用情况:%s' % gpuinfo)

    start = time.time()
    # 创建搜索器
    vs = VecSearch(dim=dim, gpu=gpu)
    # 把向量添加到搜索器
    ret = vs.add(X)
    #print(ret)
    # 添加数据后一定要索引
    vs.reindex()

    # 计算时间
    end = time.time()
    total_time = end - start
    print('创建用时:%4f秒' % total_time)

    # 查看当前进程使用的内存情况
    import os,psutil
    process = psutil.Process(os.getpid())
    print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
    
    # 进行测试
    print('单条查询测试'.center(40,'-'))
    Q = np.random.random((test_times, dim))
    Q[:, 0] += np.arange(test_times) / test_times

    q = Q[0]
    start = time.time()
    D,I = vs.search(q, top=top_n, nprobe=10)
    
    # 显示详细结果
    def showdetail (X,q,D,I):
        print('显示查询结果，并验证余弦相似度...')
        #for i in range(len(I[0])):
        for i,v in enumerate(I[0]):
            #np.squeeze(X[v])
            c = CosSim_dot(Q[0], X[v])
            r = (v, D[0][i], c) # CosSim_dot(Q[0], X[v]),
            print('索引号:%5d, 距离:%f, 余弦相似度:%f' % r )
            #rv = X[v][:10]
            #print('\n查询结果(超长只显示前10维:%s' % rv)

    showdetail (X,q,D,I)
    end = time.time()
    total_time = (end - start)*1000
    print('总用时:%d毫秒' % (total_time) )


    print('批量查询测试'.center(40,'-'))
    print('正在批量测试%d次，每次返回Top %d，请稍候...' % (test_times,top_n) )
    start = time.time()
    for i in range(test_times):
        D,I = vs.search(Q[i])
    end = time.time()
    total_time = (end - start)*1000
    print('总用时:%d毫秒, 平均用时:%4f毫秒' % (total_time, total_time/test_times) )
    #return 
        
    # 人工测试
    while 1:
        print('-'*40)
        txt = input("回车开始测试(Q退出)：").strip()
        if txt.upper()=='Q': break
      
        # 随机生成一个向量
        print('随机生成一个查询向量...')
        #q = rng.random_sample(dim)
        Q = np.random.random(dim)
        print("query:%s..." % q[:5])

        # 查询
        start = time.time()
        r = vs.search(q, top=top_n)
        end = time.time()
        print('查询结果:...')
        print('相似度:%s \n索引号:%s' % r) 
        total_time = end - start
        print('用时:%4f 毫秒' % (total_time*1000) )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='faiss速度测试工具')
    parser.add_argument('--total', default=100000, type=int, help='总数据量,默认10万') # required=True,
    parser.add_argument('--dim', default=768, type=int, help='向量维度,默认768')
    parser.add_argument('--test_times', default=10000, type=int, help='测试次数，默认1万')
    parser.add_argument('--top_n', default=5, type=int, help='每次返回条数,默认5')
    parser.add_argument('--gpu', default=-1, type=int, help='使用GPU,-1=不使用（默认），0=使用第1个，>0=使用全部')
    args = parser.parse_args()
    #print(args)
    test_task(args)
