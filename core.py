# -*- coding: utf-8 -*-
# @Time : 2022/8/23 16:04
# @Author : DENG GQ
# @Email : 2832885485@qq.com
# @File : core.py
# @Project : VG_join
# @内容:实现普通VG的fast版,实现VG序列间的相似度度量,以二者为基础实现一维时间序列间在VG视角下的子序列匹配.并优化
import abc
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numba
import multiprocessing as mlp


def _convert_2_row_vector(TS) ->np.ndarray:
    """
    将能转为行向量的转为行向量
    :param TS:时间序列
    :return:行向量版TS
    """
    if not isinstance(TS,np.ndarray):
        TS=np.asarray(TS)
    _shape = TS.shape
    if TS.ndim == 1:
        return TS.reshape([1, _shape[0]])
    if TS.ndim == 2:
        if _shape[0] == 1:
            return TS
        if _shape[1] == 1:
            return TS.T
    raise f"仅允许输入能转为行向量的时间序列,而目前输入了shape={TS.shape}"


def _fastVG(TS: np.ndarray, window: int):
    """
    内部函数,用于生成TS的有限视距的VG序列
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """

    _len = TS.shape[1]
    res = np.full((_len - 1, window - 1), False)
    max_k = np.full([1, _len], np.NINF)
    for i in range(1, window):
        k = (TS[0, i:] - TS[0, :-i]) / i
        index = k > max_k[0,:-i]
        max_k[0,np.where(index)[0]] = k[index]
        res[np.where(index)[0], i - 1] = True
    return res[:-1,1:]
def compare_two_TS(TS1,TS2):
    TS1,TS2=_convert_2_row_vector(TS1),_convert_2_row_vector(TS2)
    VG1,VG2=_fastVG(TS1,TS1.shape[1]),_fastVG(TS2,TS2.shape[1])
    return np.sum(VG1!=VG2)


class  _Abstract_model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def draw_profile(self):
        """
        画出profile图像
        :return:
        """
        pass

    @abc.abstractmethod
    def _task_unit(self):
        """
        在匹配过程中的任务单元,通过并行计算来大大优化整体效率
        :return:
        """
        pass


class _Abstract_self_join(_Abstract_model,metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def self_match(self):
        """自连接"""
        pass

    def _get_VG_of_i_th_subsequence(self, i: int):
        """
        获得该模型中TS[i:i+window]所形成的VG的变种邻接矩阵
        :param i:
        :return: 二维数组,形状为等腰直角三角形
        """
        if 0 <= i < self.subsequence_num:
            return [self.adjacency_matrix[i + self.window - 2 - k, :k] for k in range(self.window - 2, 0, -1)]
        else:
            raise IndexError(f"i:{i}越界,应为0-{self.subsequence_num}")

    def _record_distance(self, all_task_unit_res:[[]]):
        """
        将_task_unit结果存入distance_matrix
        :param i:
        :param j:
        :return:
        """
        for i,task_unit_res in enumerate(all_task_unit_res):
            for j,values in enumerate(task_unit_res):
                self.distance_matrix[j,i+1+j]=values
                self.distance_matrix[i+1+j,j]=values

    def get_MP(self):
        """
        返回Matrix Profile 矩阵轮廓
        :return:
        """
        return self.distance_matrix.min(axis=0)

    def get_argMP(self):
        """
        返回各子序列的相似子序列
        :return:
        """
        return self.distance_matrix.argmin(axis=0)

    def draw_profile(self, **kwargs):
        plt.figure(**kwargs)
        plt.plot(self.get_MP())
    def to_Matrix_Profile(self):
        """
        返回MP的数据格式
        :return:
        """
        return {
            # A numpy array of the matrix profile.
            'mp': self.get_MP(),
            # The profile index as a numpy array.
            'pi': self.get_argMP(),

            # The right matrix profile as a numpy array.
            'rmp':None,

            # The right matrix profile indices as a numpy array.
            'rpi': None,

            # The left matrix profile as a numpy array.
            'lmp':None,

            # The left matrix profile indices as a numpy array.
            'lpi': None,

            # The distance metric in the matrix profile (Euclidean or Pearson).
            'metric': "VG_disatance",

            # The window size used.
            'w': self.window,

            # The exclusion zone for non-trivial matches.
            'ez': 0,

            # A flag specifying if a self-join or similarity join was computed.
            'join': False,

            # A float from 0 to 1 indicating how many samples were taken to compute the MP.
            'sample_pct': 1,

            # The original data where ts is the time series and query is the query.
            'data': {
            'ts': self.TS.reshape(self.TS_len),
            'query': None
            },


            # This is used internally to determine what this data structure is.
            'class': "MatrixProfile",

            # The algorithm used to compute this matrix profile.
            'algorithm': "VG_join"
        }

class VG_self_joinModel(_Abstract_self_join):
    def __init__(self, window, n_core: int=1, abs_of_j_sub_i_threshold: int = None):
        """

        :param window: 窗口长度,VG的视距
        :param n_core: 可调用的CPU核心数
        :param abs_of_j_sub_i_threshold: 如果自连接匹配的话,需设置的过近的阈值,如当2个窗口之间的距离相差小于窗口长度一半时,这2窗口内的VG序列极有可能是相似(注:这是采用瓯距的MP的问题,在VG视角下不一定有问题)
        """
        self.window = window
        # self.has_created = False
        self.n_core = n_core
        self.abs_of_j_sub_i_threshold = abs_of_j_sub_i_threshold

    def self_match(self, TS: np.ndarray):
        if self.abs_of_j_sub_i_threshold is None:
            self.abs_of_j_sub_i_threshold = self.window // 2
        self.TS = _convert_2_row_vector(TS)
        self.TS_len = self.TS.shape[1]
        self.subsequence_num = self.TS_len - self.window + 1
        self.adjacency_matrix = _fastVG(self.TS, self.window)
        self.distance_matrix = np.full([self.subsequence_num, self.subsequence_num], np.inf)
        all_task_res = list(self._task_unit(0, j) for j in range(1,self.subsequence_num))
        # with mlp.Pool(self.n_core) as pool:
        #     all_task_res=pool.starmap(self._task_unit,[(0,j) for j in range(1,self.subsequence_num)])
        self._record_distance(all_task_res)
    def _task_unit(self, i: int, j: int):
        """
        从i，j处右斜向下计算，即计算(i,j),(i+1,j+1),(i+2,j+2)...，将这些位置的结果以一维数组返回
        :param i:
        :param j:
        :return:
        """
        res=[0]*(self.subsequence_num-max(i,j))
        start_i_window = self._get_VG_of_i_th_subsequence(i)
        start_j_window = self._get_VG_of_i_th_subsequence(j)
        distance_deuqe = deque([np.sum(start_i_window[k] != start_j_window[k]) for k in range(self.window - 2)],
                               maxlen=self.window - 2)
        res[0]=sum(distance_deuqe)
        for k in range(1, self.subsequence_num-max(i,j)):
            distance_deuqe.popleft()
            for t in range(self.window - 3):
                distance_deuqe[t] += self.adjacency_matrix[i + k + t, self.window - 3 - t] != self.adjacency_matrix[
                    j + k + t, self.window - 3 - t]
            distance_deuqe.append(self.adjacency_matrix[i + k + t + 1, 0] != self.adjacency_matrix[j + k + t + 1, 0])
            res[k]=sum(distance_deuqe)
        return res


class _abstractmodel(_Abstract_model):
    @abc.abstractmethod
    def alien_match(self, TS1, TS2):
        """异连接"""
        pass
# if __name__ == '__main__':
#     ts=np.array([1,4,3,1,7,20,16,18,5])
#     ts=_convert_2_row_vector(ts)
#     print(_fastVG(ts,5))
#     model=VG_self_joinModel(5)
#     model.self_match(ts)
#     print(
#         model.distance_matrix
#     )