#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""
  #    以字典的形式存储图信息(也就是邻接表)，其中key是结点的编号，value是相邻结点编号组成的list

  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    #            """返回图中的所有结点"""
    return self.keys()

  def adjacency_iter(self):
    #返回邻接表
    return self.iteritems()

  def subgraph(self, nodes={}):
    #    #"""给定顶点集合nodes，返回对于的子图"""
    subgraph = Graph()
    import pdb
    pdb.set_trace()
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]#把nodes中的节点和对应的边添加到子图subgraph中
        
    return subgraph

  def make_undirected(self):#self:Graph(<class 'list'>, {1: [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32], 2: [1, 3, 4, 8, 14, 18, 20, 22, 31], 3: [1, 2, 4, 8, 9, 10, 14, 28, 29, 33], 4: [1, 2, 3, 8, 13, 14], 5: [1, 7, 11], 6: [1, 7, 11, 17], 7: [1, 5, 6, 17], 8: [1, 2, 3, 4], 9: [1, 3, 31, 33, 34], 10: [3, 34], 11: [1, 5, 6], 12: [1], 13: [1, 4], 14: [1, 2, 3, 4, 34], 15: [33, 34], 16: [33, 34], 17: [6, 7], 18: [1, 2], 19: [33, 34], 20: [1, 2, 34], 21: [33, 34], 22: [1, 2], 23: [33, 34], 24: [26, 28, 30, 33, 34], 25: [26, 28, 32], 26: [24, 25, 32], 27: [30, 34], 28: [3, 24, 25, 34], 29: [3, 32, 34], 30: [24, 27, 33, 34], 31: [2, 9, 33, 34], 32: [1, 25, 26, 29, 33, 34], 33: [3, 9, 15, 16, 19, 21, 23, 24, 30, 31, 32, 34], 34: [9, 10, 14, 15, 16, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 32, 33]})
  
    t0 = time()

    for v in list(self):#v是有向边起点，v=1
      for other in self[v]:#self[1]=[2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32]
        if v != other:#other = 2,3,...
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))
    # pdb.set_trace()
    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))#对邻接节点去重排序
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
    # pdb.set_trace()
    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: #自己连接自己，一跳
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):#检查是否有自环
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):#检查v1、v2之间是否存在边
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False
  #
  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):#path_length，可以大于节点总数
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.#重新开始的可能性
        start: the start node of the random walk.
    """
    #指定起点start则path从start开始
      # 若是没有指定，随机指定节点为起点
      # 这里的start即为算法里的vi
    G = self#邻接矩阵
    if start:#指定随机
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]
    #当path长度还没达到限制，继续游走
      # 这里path_length即为算法里的t
    #若是当前生成的随机数大于等于alpha，则继续往下随机挑选节点走
          # 否则，从起点重新游走
    while len(path) < path_length:#path_length=40
      cur = path[-1]#路径上最新的节点
      if len(G[cur]) > 0:
        if rand.random() >= alpha:#alpha=0，肯定会从最新节点的邻接节点中随机选择一个节点进行访问
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    # pdb.set_trace()
    return [str(node) for node in path]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []
  #对每个节点进行num_paths次的随机游走
  nodes = list(G.nodes())
  # pdb.set_trace()
  for cnt in range(num_paths):#10，选择10条随机的path
    rand.shuffle(nodes)#打乱nodes
    for node in nodes:#随机选择一个节点开始，10*34条path，每条path长40
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))#path_length=40
  pdb.set_trace()
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    #permutations
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):#按n分组，不足用padvalue补充，默认用None填充，构造长度为n的
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)#返回元组列表

def parse_adjacencylist(f):
  adjlist = []
  import pdb
  # pdb.set_trace()
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist
import pdb
def parse_adjacencylist_unchecked(f):#解析邻接矩阵文件
  adjlist = []
  # pdb.set_trace()
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]]) #用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
  # pdb.set_trace()
  return adjlist#转换为二维列表，一个数组是一行

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):
  # 每chunksize个顶点的连接信息为一个chunk
  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()
  
  total = 0 
  with open(file_) as f:
    for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):#因为chunksize>文件字符长度，grouper返回的列表只有一个chunk
      adjlist.extend(adj_chunk)
      # pdb.set_trace()
      total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x)
  
  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):#adjlist二维列表
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors#构造邻接矩阵

    return G


