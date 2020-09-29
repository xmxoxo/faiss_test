# faiss安装与测试

在NLP的应用中，经常需要用到对向量的搜索，如果向量的数量级非常大，比如1千万，甚至上亿条，普通的方式就满足不了生产需要了，falcebook开源的faiss框架能够解决“海量向量搜索”的问题。
    
    在网上也有很多faiss的相关文章，大多都是介绍安装以及对官方demo的运行测试。到底faiss的速度如何，使用内存情况如何，如何把faiss包装成服务与项目结合，带着这些问题，笔者做了一些尝试，把这些过程做一下分享。

## faiss介绍
Faiss是Facebook Ai Research开发的一款稠密向量检索工具。
简单来说就是向量搜索工具。
引用Faiss Wiki上面的一段简介说明下特点：
1.Faiss是针对稠密向量进行相似性搜索和聚类的一个高效类库。
2. 它包含可搜索任意大小的向量集的算法，这些向量集的大小甚至都不适合RAM。
3. 它还包含用于评估和参数调整的支持代码。
4. Faiss用C ++编写，并且有python2与python3的封装代码。
5. 一些最有用的算法在GPU上有实现。
6. Faiss是由Facebook AI Research开发的。

## faiss安装
faiss需要在linux下安装，这里的测试环境是：
ubuntu 16.04
python 3.6.6
使用conda安装faiss是便捷的方式。
```
# 更新conda
conda update conda
# 先安装mkl
conda install mkl

# faiss提供gpu和cpu版，根据服务选择
# cpu版本
conda install faiss-cpu -c pytorch

# gpu版本 -- 记得根据自己安装的cuda版本安装对应的faiss版本，不然会出异常

# For CUDA8
conda install faiss-gpu cudatoolkit=8.0 -c pytorch 

# For CUDA9
conda install faiss-gpu cudatoolkit=9.0 -c pytorch 

# For CUDA10
conda install faiss-gpu cudatoolkit=10.0 -c pytorch 
```
注意，如果安装不成功时，可以尝试去掉 命令行最后的`-c pytorch`
```
# 校验是否安装成功
python -c "import faiss"
```
## 项目说明
本项目的源码地址为：https://github.com/xmxoxo/faiss_test

项目是为了做一个对比实验，来看一下faiss的使用及速度。
我们随机生成10万个768维的向量，加载到内存中，
然后分别用普通的暴力搜索和faiss搜索两种方式去搜索，对比搜索的平时用时。

整个实验过程的开发日志见： develop log.txt


## 普通暴力搜索
为了对比搜索的速度，先用普通的暴力搜索，就是一个个计算余弦相似度，保留TOP N。
在测试工具里增加了交互，运行后可以自行测试。

运行命令：
```
python VecSearch_force.py
```

10万条数据的运行结果：
```
root@ubuntu:/mnt/sda1/transdat/VecSearch# python VecSearch_Lib.py
===========大批量向量余弦相似度计算-[暴力版]===========
随机生成100000个向量，维度：768
正在创建搜索器...
添加用时:0.027474秒
Used Memory: 672.36328125 MB
-----------------单条查询测试-----------------
搜索结果: [0.78086126 0.77952558 0.77949381 0.77775592 0.77547914] [ 541  443 1472  370  209]
显示查询结果，并验证余弦相似度...
索引号:  541, 距离:0.780861
索引号:  443, 距离:0.779526
索引号: 1472, 距离:0.779494
索引号:  370, 距离:0.777756
索引号:  209, 距离:0.775479
-----------------批量查询测试-----------------
批量测试次数：100 次，请稍候...
总用时:93 秒, 平均用时:932.549202 毫秒
```

## faiss向量搜索

faiss向量搜索测试工具中，对faiss向量搜索进行了包装，
后续会发布了一个faiss向量搜索通用服务端。

在命令行中可以设置各类测试参数，支持GPU；

具体帮助信息可以使用以下命令查看：
```
python VecSearch_faiss.py -h
```

帮助信息如下：
```
VecSearch_faiss.py: error: argument -h/--help: ignored explicit argument 'elp'
(base) root@ubuntu:/mnt/sda1/transdat/VecSearch# python VecSearch_faiss.py -h
usage: VecSearch_faiss.py [-h] [--total TOTAL] [--dim DIM]
                          [--test_times TEST_TIMES] [--top_n TOP_N]
                          [--gpu GPU]

faiss速度测试工具

optional arguments:
  -h, --help            show this help message and exit
  --total TOTAL         总数据量
  --dim DIM             向量维度
  --test_times TEST_TIMES
                        测试次数
  --top_n TOP_N         每次返回条数
  --gpu GPU             使用GPU,-1=不使用，0=使用第1个，>0=使用全部

```

10万条768维向量搜索性能：

```
root@ubuntu:/mnt/sda1/transdat/VecSearch# python VecSearch_faiss.py
=========大批量向量余弦相似度计算-[faiss版]==========
随机生成100000个向量，维度：768
正在创建搜索器...
(0, 99999)
创建用时:2.116421秒
Used Memory: 1326.91015625 MB
-----------------单条查询测试-----------------
显示查询结果，并验证余弦相似度...
索引号: 1058, 距离:114.878273, 余弦相似度:0.771127
索引号:  541, 距离:115.051292, 余弦相似度:0.780861
索引号:  370, 距离:115.715881, 余弦相似度:0.777756
索引号:  209, 距离:115.731331, 余弦相似度:0.775479
索引号: 1472, 距离:115.832954, 余弦相似度:0.779494
总用时:9毫秒
-----------------批量查询测试-----------------
正在批量测试1000次，每次返回Top 5，请稍候...
总用时:1415毫秒, 平均用时:1.415620毫秒

```

对比截图见images目录。

