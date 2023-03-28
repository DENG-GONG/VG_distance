# -*- coding: utf-8 -*-
# @Time : 2022/9/26 10:19
# @Author : DENG GQ
# @Email : 2832885485@qq.com
# @File : _tools.py
# @Project : Project
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.sparse import lil_matrix
import networkx as nx

__all__=['convert_2_row_vector', 'fastWVG_to_Graph', 'from_WVG_graph_to_3_distance','fastVG_to_adj','from_VG_adj_to_vg_and_degree_distance']
def convert_2_row_vector(TS) ->np.ndarray:
    """
    将能转为行向量的转为行向量
    :param TS:时间序列
    :return:行向量版TS
    """
    if not isinstance(TS,np.ndarray):
        try:
            TS=np.asarray(TS)
        except :
            raise "输入的参数无法转为ndarray"
    if TS.ndim == 1:
        return TS.reshape([-1,1])
    elif TS.ndim == 2:
        return TS.copy()
    else:
        raise f"仅允许输入能转为行向量的时间序列,而目前输入了shape={TS.shape}"
def fastVG_to_Graph(TS: np.ndarray)->nx.Graph:
    """
    内部函数,用于生成TS的有限视距的VG序列
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """

    _len = TS.shape[0]
    if TS.shape[1]!=1:
        raise "VG仅允许n*1的矩阵输入"
    G=nx.Graph()
    G.add_nodes_from(range(_len))
    # res = np.full((_len - 1, window - 1), False)
    max_k = np.full([ _len], np.NINF)
    for i in range(1, _len):
        k = (TS[i:,0] - TS[:-i,0]) / i
        index = k > max_k[:-i]
        max_k[np.where(index)[0]] = k[index]
        G.add_edges_from([(j,j+i) for j in np.where(index)[0]])
        # res[np.where(index)[0], i - 1] = True
    return G
def fastWVG_to_Graph(TS: np.ndarray)->nx.Graph:
    """
    用于生成TS的VG网络Graph实列
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """
    TS=convert_2_row_vector(TS)
    _len = TS.shape[0]

    G=nx.Graph()
    G.add_nodes_from(range(_len))
    max_k = np.full([_len], np.NINF)
    for i in range(1, _len):
        k = (TS[i:,0] - TS[:-i,0]) / i
        index = k > max_k[:-i]
        max_k[np.where(index)[0]] = k[index]
        G.add_edges_from([(j+i,j) for j in np.where(np.all([k>=0,index],axis=0))[0]],type=1)
        G.add_edges_from([(j+i,j) for j in np.where(np.all([k<0,index],axis=0))[0]],type=-1)

        # res[np.where(np.all([k>=0,index],axis=0))[0], i - 1] = 1
        # res[np.where(np.all([k<0,index],axis=0))[0], i - 1] = -1

    return G
def fastWVG_to_adj(TS:np.ndarray):
    """
    用于生成TS的VG网络邻接矩阵
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """
    TS=convert_2_row_vector(TS)
    _len = TS.shape[0]
    res=np.zeros((_len,_len),dtype=int)
    max_k = np.full([_len], np.NINF)
    for i in range(1, _len):
        k = (TS[i:,0] - TS[:-i,0]) / i
        index = k > max_k[:-i]
        max_k[np.where(index)[0]] = k[index]
        pos_index=np.where(np.all([k>=0,index],axis=0))[0]
        neg_index=np.where(np.all([k<0,index],axis=0))[0]

        res[pos_index,pos_index+i]=1
        res[neg_index,neg_index+i]=-1
    res+=res.T
    # print(res)
    return res
def fastVG_to_adj(TS:np.ndarray):
    """
    用于生成TS的VG网络邻接矩阵
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """
    TS=convert_2_row_vector(TS)
    _len = TS.shape[0]
    res=np.zeros((_len,_len),dtype=int)
    max_k = np.full([_len], np.NINF)
    for i in range(1, _len):
        k = (TS[i:,0] - TS[:-i,0]) / i
        index = k > max_k[:-i]
        index_arg=np.where(index)[0]
        max_k[index_arg] = k[index]
        res[index_arg,index_arg+i]=1

    res+=res.T
    # print(res)
    return res

def from_WVG_graph_to_3_distance(WVG1:nx.Graph,WVG2:nx.Graph):
    """从WVG的graph生成3种距离"""
    def form_sorted_v_u_d(v,u,d):
        l=sorted([v,u])
        l.append(d['type'])
        return l
    _len=len(WVG1)
    edgs1,edgs2=sorted([form_sorted_v_u_d(v,u,d) for v,u,d in list(WVG1.edges(data=True))]),sorted([form_sorted_v_u_d(v,u,d) for v,u,d in list(WVG2.edges(data=True))])
    # print(edgs1,edgs2)
    degree1=[x[1] for x in WVG1.degree]
    degree2=[x[1] for x in WVG2.degree]
    degree_dis=1-abs(pearsonr(degree1,degree2)[0])
    vg_sim=0
    wvg_sim=0
    len1,len2=len(edgs1),len(edgs2)
    i,j=0,0
    while i<len1 or j<len2:
        vi,ui,di=edgs1[i]
        vj,uj,dj=edgs2[j]
        if vi<vj:
            i+=1
        elif vi>vj:
            j+=1
        else:
            if ui<uj:
                i+=1
            elif ui>uj:
                j+=1
            else:
                vg_sim+=1
                if di==dj:
                    wvg_sim+=1
                i+=1
                j+=1
    return degree_dis, 1- vg_sim / (_len**2-_len)/2, 1- wvg_sim/(_len**2-_len)/2

def from_WVG_adj_to_3_distance(WVG1:np.ndarray,WVG2:np.ndarray):
    """依次返回degree_dis,vg_dis,wvg_dis"""
    _len=WVG1.shape[0]
    wvg_dis=np.sum(WVG1!=WVG2)
    VG1,VG2=np.abs(WVG1),np.abs(WVG2)
    vg_dis=np.sum(VG1!=VG2)

    degree1,degree2=np.squeeze(np.sum(VG1,axis=1)),np.squeeze(np.sum(VG2,axis=1))

    degree_dis=euclidean(degree1,degree2)
    return degree_dis, vg_dis,wvg_dis
def from_VG_adj_to_vg_and_degree_distance(VG1:np.ndarray,VG2:np.ndarray):
    """依次返回degree_dis,vg_dis"""
    _len=VG1.shape[0]
    vg_dis=np.sum(VG1!=VG2)
    degree1,degree2=np.squeeze(np.sum(VG1,axis=1)),np.squeeze(np.sum(VG2,axis=1))

    degree_dis=euclidean(degree1,degree2)
    return degree_dis, vg_dis
def VG_dis_between_two_TS(TS1, TS2):
    """
    等长序列间的VG距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    VG1,VG2= fastVG_to_Graph(TS1), fastVG_to_Graph(TS2)
    return np.sum(VG1!=VG2)
def WVG_dis_between_two_TS(TS1, TS2):
    """
    等长序列间的VG距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    VG1,VG2= fastWVG_to_Graph(TS1), fastWVG_to_Graph(TS2)
    return np.sum(VG1!=VG2)
def degree_dis_between_two_TS(TS1,TS2):
    """
    等长序列间的VG度序列距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    VG1,VG2= fastVG_to_Graph(TS1), fastVG_to_Graph(TS2)
    return euclidean(np.sum(VG1,axis=1),np.sum(VG2,axis=1))
def norm_adjust_to_new_adjust(norm_matrix:np.ndarray):
    m=norm_matrix.shape[0]
def new_adjust_to_norm_adjust(new_matrix:np.ndarray):
    m=new_matrix.shape[0]+1
    res=np.ones([m,m],dtype=np.bool)
    for i in range(m):
        for j in range(i-1):
            try:
                v=new_matrix[j,i-j]
                res[i,j]=v
                res[j,i]=v
            except:
                print((i,j),(j,i-j))
    return res
