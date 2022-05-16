import torch
import os
import numpy as np
import random
# from scipy.sparse import coo_matrix
import scipy.sparse as sp
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    with open('./data/WN18RR/entity2id.txt', 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open('./data/WN18RR/relation2id.txt', 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

#
def load_data1(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data)

def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()
    # num = len(lines)
    # this is list for relation triples
    alpha = 0.5
    triples_data = []
    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # 对于稀疏张量，rows 列表包含稀疏张量的对应行，cols 列表包含稀疏张量的对应列. data包含关系类型
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    # 实体的邻接矩阵是无向的，源实体和尾实体应该知道它们所连接的关系类型
    adj = []
    # unique_entities = set()
    for i in relation2id.values():     # 取关系字典里的values，最外层for循环
        rows, cols, data = [], [], []
        unique_entities = set()    # 创建集合（set）是一个无序的不重复元素序列。保存训练集的头尾/实体
        for line in lines:
            e1, relation, e2 = parse_line(line)  # 将三元组按空格进行切分
            if(relation2id[relation] == i):
                unique_entities.add(e1)
                unique_entities.add(e2)
                triples_data.append(             # 保存将三元组转换为id索引的形式保存  例如：[(0,0,1)]
                    (entity2id[e1], relation2id[relation], entity2id[e2]))
                if not directed:
                    # Connecting source and tail entity
                    rows.append(entity2id[e1])
                    cols.append(entity2id[e2])
                    if is_unweigted:
                        data.append(1)
                    else:
                        data.append(relation2id[relation]+alpha)

                # Connecting tail and source entity
                rows.append(entity2id[e2])    # rows 列表保存尾实体  索引
                cols.append(entity2id[e1])    # cols 列表保存头实体  索引
                if is_unweigted:
                    data.append(1)
                else:
                    data.append(relation2id[relation]+alpha)  # data 列表保存关系 索引
        rows=np.array(rows)
        cols=np.array(cols)
        data=np.array(data)
        g = sp.coo_matrix((data, (rows, cols)), shape=(40943, 40943),dtype=np.float16)
        # g = normalize_adj(g)
        g = normalize_adj(g + sp.eye(g.shape[0]))
        g = torch.FloatTensor(np.array(g.todense()))
        # indices = torch.LongTensor([rows, cols]).to(device)
        # values = torch.FloatTensor(data).to(device)
        # sparse_matrix = torch.sparse_coo_tensor(indices, values, [40943, 40943])
        # adj.append(sparse_matrix)
        # g= (rows, cols, data)
        print('***')
        adj.append(g)



        print("number of unique_entities ->", len(unique_entities))

    return triples_data, adj
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def build_data(path='./data/WN18RR/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    #
    train_triples, train_adjacency_mat = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat = load_data1(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat = load_data1(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat),entity2id, relation2id, headTailSelector
    # return (train_triples, train_adjacency_mat), \ entity2id, relation2id, headTailSelector
