# -*- coding: utf-8 -*-
# @Time : 2023/3/21 21:02
# @Author : DENG GQ
# @Email : 2832885485@qq.com
# @File : compare_ex.py
# @Project : Visibility-Graph_distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw import dtw as dtw
from _tools import *
from scipy.spatial.distance import minkowski
import os
import multiprocessing as mlp
from scipy.spatial.distance import euclidean

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#准备要复用的函数

def compare_different_method(TS1:np.ndarray,TS2:np.ndarray,new_TS2:np.ndarray):
    """依次返回eu,vg,ma,degree"""
    # nonlocal eu_temp,vg_temp,degree_temp,manha_temp
    # TS11,TS22,TS33=convert_2_row_vector(TS1),convert_2_row_vector(TS2),convert_2_row_vector(new_TS2)
    VG1,VG2,VG3=fastVG_to_adj(TS1),fastVG_to_adj(TS2),fastVG_to_adj(new_TS2)
    # print(VG1,VG2,VG3)
    degree_dis1,vg_dis1=from_VG_adj_to_vg_and_degree_distance(VG1,VG2)
    degree_dis2,vg_dis2=from_VG_adj_to_vg_and_degree_distance(VG1,VG3)
    # vg_dis1=VG_dis_between_two_TS(TS1,TS2)
    # vg_dis2=VG_dis_between_two_TS(TS1,new_TS2)

    # wvg_dis1=WVG_dis_between_two_TS(TS1,TS2)
    # wvg_dis2=WVG_dis_between_two_TS(TS1,new_TS2)

    eu_dis1=minkowski(TS1,TS2,2)
    eu_dis2=minkowski(TS1,new_TS2,2)

    manha_dis1=minkowski(TS1,TS2,1)
    manha_dis2=minkowski(TS1,new_TS2,1)

    # degree_dis1=degree_dis_between_two_TS(TS1,TS2)
    # degree_dis2=degree_dis_between_two_TS(TS1,new_TS2)

    # abs(wvg_dis2-wvg_dis1+1)/(wvg_dis1+1)
    return [abs(eu_dis2-eu_dis1)/(eu_dis1+1),abs(vg_dis2-vg_dis1)/(vg_dis1+1),abs(manha_dis1-manha_dis2)/(manha_dis1+1),abs(degree_dis1-degree_dis2)/(degree_dis1+1)]
    # eu_temp.append(abs(eu_dis2-eu_dis1+1)/(eu_dis1+1))
    # vg_temp.append(abs(vg_dis2-vg_dis1+1)/(vg_dis1+1))
    # wvg_temp.append(abs(wvg_dis2-wvg_dis1+1)/(wvg_dis1+1))
    # manha_temp.append(abs(manha_dis1-manha_dis2+1)/(manha_dis1+1))
    # degree_temp.append(abs(degree_dis1-degree_dis2+1)/(degree_dis1+1))
def form_norm_noise_test_Data(m,db):
    """生成正太噪声干扰,db应从[0,200]对应[0,0.2]"""
    db=db/1000
    query=np.random.randn(m)
    b=np.random.randn(m)
    c=np.random.normal(0,db,m)
    b2=b+c
    return query,b,b2
def form_unif_noise_test_Data(m,db):
    """生成均匀噪声干扰,db应从[0,200]对应[0,0.2]"""
    db=db/1000
    query=np.random.randn(m)
    b=np.random.randn(m)

    c=np.random.uniform(-db,db,m)
    b2=b+c
    return query,b,b2
def form_amplitude_excursion_test_Data(m,db):
    """
    振幅偏移,db应从[-200,200]对应[-0.2,0.2]
    :param m:
    :param db:
    :return:
    """
    db=db/1000
    query=np.random.randn(m)
    b=np.random.randn(m)
    b2=b+db
    return query,b,b2
def form_Amplitude_stretch_test_Data(m,db):
    """振幅伸缩,db应从[-200,200]对应[-0.8,1.2]
    """
    db=db/1000
    query=np.random.randn(m)
    b=np.random.randn(m)
    b2=b*(1+db)
    return query,b,b2
def form_Linear_drift_test_Data(m,db):
    """
    线性漂移,db应从[-200,200]对应[-0.2,0.2]
    :param m:
    :param db:
    :return:
    """
    db=db/1000

    query=np.random.randn(m)
    b=np.random.randn(m)
    b2=b+np.arange(m)*db
    return query,b,b2
# def form_Discontinuous_test_Data(m,db):
def plot_res(all_res,xlabel,title,savename,xlim=None,ylim=None,xtick=None):
    labels=["EU","VG","Manhattan","Degree"]
    plt.figure()
    if xtick is None:
        for i,data in enumerate(all_res):
            plt.plot( data, label=labels[i], linewidth=3 if i == 1 else 2)
    else:
        for i, data in enumerate(all_res):
            plt.plot(np.arange(*xtick),data,label=labels[i],linewidth=3 if i==1 else 2)

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('avg of abs(dis2-dis1)/(dis1+1)')
    plt.title(title)
    plt.savefig(f'fig/{savename}.png',dpi=300)
    plt.savefig(f'fig/svg/{savename}.svg',dpi=300)
    # plt.show()

# xlabels=['序列长度','噪声强度','序列长度','噪声强度','序列长度','偏移强度','序列长度','伸缩强度','序列长度','漂移强度']
# ylabels=['正太分布的噪声-m','均匀分布的噪声','正态分布的噪声','均匀分布的噪声',"",'','','','','']
savenames2=["m-正太",'db-正太','m-均匀','db-均匀','m-抗偏','db-抗偏','m-抗伸','db-抗伸','m-抗漂','db-抗漂']
# savenames=["抗噪-m-正太-细节",'抗噪-m-均匀-细节','抗噪-db-正太-细节','抗噪-db-均匀-细节','抗偏-m-细节','抗偏-db-细节','抗伸-m-细节','抗伸-db-细节','抗漂-m-细节','抗漂-db-细节']
# title=["讨论不同长度的序列稳定性对影响","讨论不同长度的序列稳定性对影响","讨论不同强度的正太噪声对稳定性的影响","讨论不同强度的均匀噪声对稳定性的影响"]
def m_shiyan(i,len_,form_noise_fun,remake=False,iter=1000,xlim=[4,200],ylim=[0,0.2]):
    i=i*2
    if not remake and os.path.exists(f"save_data/compare/mat/{savenames2[i]}.npy"):
        all_res=np.load(f"save_data/compare/mat/{savenames2[i]}.npy")
    else:
        all_res=np.zeros([4,len_])
        for m in range(4,len_):
            all_temp=np.zeros([4,iter])
            for k in range(iter):
                all_temp[:,k]=compare_different_method(*form_noise_fun(m,np.random.randint(1,200)))
                # print(all_temp[:,k])
            all_res[:,m]=np.mean(all_temp,axis=1)
        os.makedirs(f"save_data/compare/mat", exist_ok=True)
        np.save(f"save_data/compare/mat/{savenames2[i]}", all_res)
    plot_res(all_res,"序列长度",savename=savenames2[i],xlim=xlim,ylim=ylim,title="讨论不同长度的序列对不稳定性对影响")
    print(f"{savenames2[i]} have done")
def db_shiyan(i, start, end, form_noise_fun, remake=False, iter=1000, ylim=[0, 0.2]):
    i=2*i+1
    if not remake and os.path.exists(f"save_data/compare/mat/{savenames2[i]}.npy"):
        all_res=np.load(f"save_data/compare/mat/{savenames2[i]}.npy")
    else:
        all_res=np.zeros([4, (end - start)])
        all_res[:,0]=0
        for index,j in enumerate(range(start, end + 1)):
            all_temp=np.zeros([4,iter])
            for k in range(iter):
                all_temp[:,k]=compare_different_method(*form_noise_fun(np.random.randint(4,200),j))
                # print(all_temp[:,k])
            all_res[:,index]=np.mean(all_temp,axis=1)
        os.makedirs(f"save_data/compare/mat", exist_ok=True)
        np.save(f"save_data/compare/mat/{savenames2[i]}", all_res)
    plot_res(all_res,"干扰强度", savename=savenames2[i], ylim=ylim, title="讨论不同强度的干扰对不稳定性的影响",xtick=[start,end+1])
    print(f"{savenames2[i]} have done")
def re_plot(i,xlabel,title,xlim,ylim):
    mat=np.load(f"save_data/compare/mat/{savenames2[i]}.npy")
    labels=["EU","VG","Manhattan","Degree"]
    plt.figure()
    for j,data in enumerate(mat):
        plt.plot(data,label=labels[j],linewidth=4 if j==1 else 2)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('avg of abs(dis2-dis1)/(dis1+1)')
    plt.title(title)
    plt.savefig(f'fig/{savenames2[i]}_{ylim[1]}.png')
    plt.savefig(f'fig/svg/{savenames2[i]}_{ylim[1]}.svg')
    # plt.show()
def db_shiyan_exp(start,end,form_noise_fun,iter=200,chusu=1000):
    all_res=np.zeros([4,end+1])
    # all_res[:,0]=0
    for j in range(start-1,end+1):
        all_temp=np.zeros([4,iter])
        for k in range(iter):
            temp_res=compare_different_method(*form_noise_fun(np.random.randint(4,200),j/chusu))
            all_temp[:,k]=temp_res
            # print(all_temp[:,k])
        all_res[:,j]=np.mean(all_temp,axis=1)
        # if np.random.random()<0.1:
        #     print(all_temp[1,:20])
    labels=["EU","VG","Manhattan","Degree"]
    plt.figure()
    for j,data in enumerate(all_res):
        plt.plot(data,label=labels[j],linewidth=3 if j==1 else 2)
    plt.xlim(start,end)
    plt.ylim(0,0.2)
    plt.legend()
    # plt.xlabel(xlabel)
    plt.ylabel('avg of abs(dis2-dis1)/(dis1+1)')
    plt.show()
    # plt.title(title)
    # plt.savefig(f'fig/{savename}.png')
    # plt.savefig(f'fig/svg/{savename}.svg')
    # os.makedirs(f"save_data/compare/mat",exist_ok=True)
    # np.save(f"save_data/compare/mat/{savenames2[i]}",all_res)
    # plot_res(all_res,"干扰强度",savename=savenames2[i],xlim=xlim,ylim=ylim,title="讨论不同强度的干扰对稳定性的影响")
    # print(f"{savenames2[i]} have done")
if __name__=="__main__":
    shi_yan_list=[form_norm_noise_test_Data,form_unif_noise_test_Data,form_amplitude_excursion_test_Data,form_Amplitude_stretch_test_Data,form_Linear_drift_test_Data]
    pool=mlp.Pool(5)
    pool.starmap_async(m_shiyan,zip(range(5),[200]*5,shi_yan_list))
    pool.starmap_async(db_shiyan,zip(range(2),[0]*2,[200]*2,shi_yan_list[:2]))
    pool.starmap(db_shiyan,zip(range(2,5),[-200]*3,[200]*3,shi_yan_list[2:] ))
    pool.close()
    pool.join()
    # re_plot(6,"序列长度","讨论不同长度的序列对稳定性的影响",[4,200],[0,0.3])
    # re_plot(8,"序列长度","讨论不同长度的序列对稳定性的影响",[4,200],[0,0.5])
    # re_plot(7,"干扰强度","讨论不同强度的干扰对稳定性的影响",[0,200],[0,0.5])
    # a,b,c=form_unif_noise_test_Data(10,0.3)
    # print(c-b)
    # for i in range(5):
    #     db_shiyan(i,300,shi_yan_list[i])
    #     res=np.load(f"save_data/compare/mat2/{savenames2[2 * i - 1]}.npy")
    #     np.save(f"save_data/compare/mat/{savenames2[2 * i + 1]}.npy",res)
    # db_shiyan_exp(0,300,form_Linear_drift_test_Data,iter=10)
    # TS1,TS2,new_TS2=form_Linear_drift_test_Data(10,0.02)
    # VG1,VG2,VG3=fastVG_to_adj(TS1),fastVG_to_adj(TS2),fastVG_to_adj(new_TS2)
    # print(VG1,VG2,VG3)
    # degree1,degree2,degree3=np.squeeze(np.sum(VG1,axis=1)),np.squeeze(np.sum(VG2,axis=1)),np.squeeze(np.sum(VG3,axis=1))
    # degree_dis1,vg_dis1=from_VG_adj_to_vg_and_degree_distance(VG1,VG2)
    # degree_dis2,vg_dis2=from_VG_adj_to_vg_and_degree_distance(VG1,VG3)
    # print(abs(vg_dis2-vg_dis1)/(vg_dis1+1))
#     m_shiyan(4,200,form_Linear_drift_test_Data,True,ylim=None)
