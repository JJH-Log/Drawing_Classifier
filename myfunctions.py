#!/usr/bin/env python
# coding: utf-8

# ## 0. 라이브러리

# In[12]:


import os
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shutil
import pickle
from tqdm import tqdm
from colorsys import rgb_to_hsv
from keras.preprocessing.image import load_img

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

from PIL import Image

from numpy import dot
from numpy.linalg import norm
import bisect


# ## 1. 이미지 파일명 불러오기

# In[3]:


def get_file_names(img_path, ends, sample_ratio = 1):
    file_names = []

    with os.scandir(img_path) as files:
        for file in files:
            # .jpg .png 등 확장자명이 'g'로 끝나는 파일들 모두 읽기
            if file.name.endswith(ends):
                file_names.append(img_path + file.name)

    if sample_ratio < 1:
        file_names = random.sample(file_names, int(len(file_names) * sample_ratio))
        
    return file_names


# ## 2. 데이터 추출하기

# ### 2-1. HSV flatten data

# In[1]:


def extract_data_hsv_flatten(file_name, img_size = (360, 360), img_grid = 5, clr_grid = (12, 8, 8)):
    img = cv2.imread(file_name)
    img = cv2.resize(img, img_size)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    grid_count = np.zeros([img_grid * 2, clr_grid[0] * clr_grid[1] * clr_grid[2]])
    clr_data = []
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            newH = (int)((img_hsv[i][j][0] / 180) * clr_grid[0]) #range of H : 0 ~ 179
            newS = (int)((img_hsv[i][j][1] / 256) * clr_grid[1]) #range of S : 0 ~ 255
            newV = (int)((img_hsv[i][j][2] / 256) * clr_grid[2]) #range of V : 0 ~ 255
            label = newH * clr_grid[1] * clr_grid[2] + newS * clr_grid[2] + newV
            grid_count[(int)((i / img_size[0]) * img_grid)][label] += 1
            grid_count[img_grid + (int)((i / img_size[0]) * img_grid)][label] += 1
    for i, grid in enumerate(grid_count):
        grid_total = grid_count[i].sum()
        clr_data.append(grid_count[i] / grid_total) 
    return np.array(clr_data)


# ### 2-2. RGB-HSV data

# In[7]:


#tf.rgb_to_hsv version (구버전)
def extract_data_rgbhsv_before(file_name, img_size = (360, 360)):
    img = load_img(file_name, target_size = img_size)
    img = np.array(img)
    img = img.reshape(img_size[0] * img_size[1], 3)
    hsv_data = []
    for pixel in img:
        hsv_data.append(rgb_to_hsv(pixel[0] / 255, pixel[1] / 255, pixel[2] / 255))
    return np.concatenate((img / 255, hsv_data), axis = 1)


# In[8]:


def extract_data_rgbhsv(file_name, img_size = (360, 360)):
    img = cv2.imread(file_name)
    img = cv2.resize(img, img_size) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(img_size[0] * img_size[1], 3)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(img_size[0] * img_size[1], 3)
    return np.concatenate((img_rgb, img_hsv), axis = 1)


# ### 2-3. HSV-YUV data

# In[9]:


def extract_data_hsvyuv(file_name, img_size = (360, 360)):
    img = cv2.imread(file_name)
    img = cv2.resize(img, img_size)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(img_size[0] * img_size[1], 3)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).reshape(img_size[0] * img_size[1], 3)
    return np.concatenate((img_hsv, img_yuv), axis = 1)


# ## 3. 클러스터링 데이터

# ### 3-1. Labels

# In[10]:


def get_clu_labels(data, cluster = 25, state = 2021):
    km = KMeans(n_clusters = cluster, random_state = state)
    km.fit(data)
    return km.labels_


# ### 3-2. Visualization with each cluster

# In[8]:


def show_img_with_clusters(file_name, labels, img_size = (360, 360), grid_size = (5, 5), clu_color = (0, 255, 0)):
    plt.figure(figsize = (25, 25))
    img = load_img(file_name, target_size = img_size)
    img = np.array(img)
    
    for idx in range(grid_size[0] * grid_size[1]):
        img_now = img.copy()
        new_labels =  np.array(list(map(lambda x: x == idx, labels))).reshape(img_size)
        for i, row in enumerate(img):
            for j, col in enumerate(row):
                if new_labels[i][j]:
                    img_now[i][j] = clu_color
        plt.subplot(grid_size[1], grid_size[0] ,idx + 1);
        plt.imshow(img_now)
        plt.axis('off')


# ### 3-3. Visualization with grid

# In[3]:


def show_img_with_grid(file_name, labels, img_size = (360, 360), grid = (8, 8), cluster = 25, ratio = 0.1, clu_color = (0, 255, 0)):
    plt.figure(figsize = (25, 25))
    img = load_img(file_name, target_size = img_size)
    img = np.array(img)
    new_labels =  labels.reshape(img_size)
    grid_count = np.zeros([grid[0] * grid[1], cluster])
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            grid_num = (int)(i / (img_size[0] / grid[0])) * grid[1] + (int)(j / (img_size[1] / grid[1]))
            grid_count[grid_num][new_labels[i][j]] += 1
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            grid_num = (int)(i / (img_size[0] / grid[0])) * grid[1] + (int)(j / (img_size[1] / grid[1]))
            if grid_count[grid_num][new_labels[i][j]] < (int)(img_size[0] * img_size[1] / grid[0] / grid[1] * ratio):
                img[i][j] = clu_color
    plt.imshow(img)
    plt.axis('off')   


# ## 4. 클러스터 데이터 추출

# ### 4.1. 클러스터 RGB평균, 중심 좌표, 평균 거리, 평균 표준편차, 비율

# In[6]:


def grouping_clusters(file_name, labels, img_size = (360, 360), cluster = 25):
    img = cv2.imread(file_name)
    img = cv2.resize(img, img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(img_size[0] * img_size[1], 3)
    
    rgbs = np.array([img_rgb[labels == i].mean(axis = 0) for i in range(cluster)])
    
    center_info = np.zeros([cluster, 3])
    new_labels = labels.reshape(img_size[0], img_size[1])
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            tar = new_labels[i][j]
            center_info[tar] += [i, j, 1]    
    centers = np.array([[x / z,y / z] for [x,y,z] in center_info])
    ratios = np.array([[z / (img_size[0] * img_size[1])] for [x,y,z] in center_info])
    
    dist_info = [[] for i in range(cluster)]
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            tar = new_labels[i][j]
            dist_info[tar].append(norm(centers[tar] - [i, j]))
    dists = np.array([[np.mean(data), np.std(data)] for data in dist_info])

    return np.concatenate((rgbs, centers, dists, ratios), axis = 1)


# ### 4-2. Visualization with cluster quality

# In[ ]:


def show_img_with_cluster_quality(file_name, labels, clu_data, img_size = (360, 360)):
    plt.figure(figsize = (25, 25))
    img = load_img(file_name, target_size = img_size)
    img = np.array(img)
    
    new_labels =  labels.reshape(img_size)
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            img[i][j] = clu_data[new_labels[i][j]][0 : 3]
    plt.imshow(img)
    plt.axis('off')


# ### 4-3. Visualization with color ratio

# In[10]:


def show_clu_by_ratio(clu_data):
    plt.figure(figsize = (8, 8))
    clu_data_sorted = clu_data[np.argsort(clu_data[:, 7])]
    plt.barh(range(len(clu_data)), clu_data_sorted[:, 7], color = clu_data[:, 0:3].astype(int)/256)
    plt.show()


# ### 4-4. Visualization with color position

# In[11]:


def show_clu_by_pos(clu_data):
    plt.figure(figsize = (8, 8))
    plt.xlim([0, 360])
    plt.ylim([0, 360])
    plt.gca().invert_yaxis()
    plt.scatter(x = clu_data[:, 4], y = clu_data[:, 3], s= clu_data[:,7] * 10000, c = clu_data[:, 0:3].astype(int)/256)
    for i in range(len(clu_data)):
        plt.text(x = clu_data[i, 4] - 1.2, y = clu_data[i, 3] + 0.4, s = str(i), size = 'small')
    plt.show()


# ## 5. 데이터 저장

# In[ ]:


def save_clu_data(save_path, file_name, data):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except OSError:
        print ('Error: Creating directory. ' +  save_path)
    try:
        save_name = save_path + file_name.rsplit('.')[0].rsplit('/')[-1] + ".npy"
        np.save(save_name, data)
    except:
        print ('Error: Creating Data. ' + save_name)


# ## 1~5. Extract Color

# In[9]:


def extract_color_v1(img_path, save_path, img_size = (360, 360), clu = 25):
    file_names = get_file_names(img_path, 'jpg', 1)
    for target_name in tqdm(file_names):
        save_name = save_path + target_name.rsplit('.')[0].rsplit('/')[-1] + ".npy"
        if not os.path.isfile(save_name):
            hsvyuv_data = extract_data_hsvyuv(target_name, img_size = img_size)
            data_labels = get_clu_labels(hsvyuv_data, cluster = clu)
            clu_data = grouping_clusters(target_name, data_labels, cluster = clu)
            save_clu_data(save_path, target_name, clu_data)


# ## 6. 클러스터 데이터 로드

# In[ ]:


def load_clu_data(data_path, sample_ratio = 1):
    clu_data = []

    with os.scandir(data_path) as files:
        for file in files:
            if file.name.endswith('npy'):
                clu_data.append(np.load(file))
    if sample_ratio < 1:
        clu_data = random.sample(clu_data, int(len(clu_data) * sample_ratio))
        
    return np.array(clu_data)


# ## 7. 데이터 평면화 작업

# In[1]:


def flatten_clusters_rgb(clu_data, div = 5):
    flt_data = []
    for feature in clu_data: 
        R = ((feature[:,0] / 256) * div).astype(int)
        G = ((feature[:,1] / 256) * div).astype(int)
        B = ((feature[:,2] / 256) * div).astype(int)
        tar = G * div * div + R * div + B
        new_row = np.zeros(div ** 3)
        for num, val in zip(tar, feature[:, -1]):
            new_row[num] += val
        flt_data.append(new_row)
    return np.array(flt_data)


# ## 8. 차원 축소

# In[ ]:


def reduction_feature_pca(flt_data, threshold = 0.99):
    pca = PCA(n_components = len(flt_data[0]), random_state = 2021)
    pca.fit_transform(flt_data)
    
    ratio = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(ratio >= threshold) + 1
    
    pca = PCA(n_components = d, random_state = 2021)
    return pca.fit_transform(flt_data)


# ## 9. 작품 클러스터링 및 그룹화

# ### 9-1. Labels : 3-1과 동일
# - get_file_names(path, ends, sample_ratio = 1)
# - get_clu_labels(data, cluster = 25, state = 2021)

# ### 9-2. Visualization with groups

# In[ ]:


def show_img_by_group(img_path, labels, group_num, grid = (7, 3), img_size = (360, 360)):
    plt.figure(figsize = (grid[0] * 3, grid[1] * 3))
    group = []
    for path, label in zip(img_path, labels):
        if label == group_num:
            group.append(path)
    
    print(f"Cluster size : {len(group)}")
    for idx in range(min(grid[1] * grid[0], len(group))):
        
        plt.subplot(grid[1], grid[0] ,idx+1);
        img = load_img(group[idx], target_size = img_size)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')


# ## 10. 작품 저장

# In[ ]:


def save_groups(save_path, img_path, labels):
    for img, label in zip(img_path, labels):
        result_path = save_path + str(label) + "/"
        try:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
        except OSError:
            print ('Error: Creating directory. ' +  result_path)   
        shutil.copyfile(img, result_path + img.rsplit('/')[1])


# ## 6~10. Clustering Images

# In[4]:


def clustering_images_v1(data_path, img_path, result_path, clu):
    clu_data = load_clu_data(data_path)
    flt_data = flatten_clusters_rgb(clu_data)
    pca_data = reduction_feature_pca(flt_data)
    img_names = get_file_names(img_path, 'jpg')
    data_labels = get_clu_labels(pca_data, cluster = clu)
    save_groups(result_path, img_names, data_labels)


# ## 11. 특징맵 생성 작업

# ### 11-1. Cosine Simillarity

# In[2]:


def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))


# ### 11-2. Feature map with Cosine Similarity

# In[3]:


def feature_map_cossim(data):
    N = len(data)
    fmap = np.zeros([N, N])
    for i in tqdm(range(N)):
        for j in range(N):
            fmap[i][j] = cos_sim(data[i], data[j])
    return fmap


# ### 11-3. Feature map scailing with Square Function

# In[ ]:


def square_filter(val, m, M, reverse = False):
    if reverse:
        return ((val - M) ** 2) / ((m - M) ** 2)
    else:
        return ((val - m) ** 2) / ((m - M) ** 2)


# In[ ]:


def feature_map_dist_scaled(data, sigma = 0.5):
    N = len(data)
    for i in range(N):
        for j in range(i, N):
            data[j][i] = data[i][j]
    for i in range(N):
        data[i][i] = np.unique(data[i])[1] - sigma * np.std(data[i])
    return np.array([square_filter(nn, nn.min(), nn.max(), reverse = True) for nn in data])

# ### 11-4. Feature map with Cluster Matching based on MCMF

# #### 11-4-1. Init

# In[5]:


def network_flow_init(clu = 25):
    global capacity, flow, cost, adj, INF, source, sink
    v = 2 * clu + 2
    capacity = [[0]*v for _ in range(v)] # 용량
    flow = [[0]*v for _ in range(v)] # 유량
    cost = [[0]*v for _ in range(v)] # 비용
    adj = [[] for _ in range(v)] # 인접 그래프
    INF = 9876543210
    source = 2 * clu
    sink = 2 * clu + 1

    for i in range(clu):
        adj[source].append(i)
        adj[i].append(source)
        capacity[source][i] = 1
    for i in range(clu, 2 * clu):
        adj[sink].append(i)
        adj[i].append(sink)
        capacity[i][sink] = 1
    for i in range(clu):
        for j in range(clu, 2 * clu):
            adj[i].append(j)
            adj[j].append(i)
            capacity[i][j] = 1


# #### 11-4-2. Cost Function

# In[6]:


def get_cost(now, tar):
    A = 3 * norm(now[0:3] - tar[0:3]) # RGB 차이
    B = norm(now[3:5] - tar[3:5]) # 중심 좌표 차이
    C = 2 * norm(now[5:7] - tar[5:7]) # 평균, 분산 차이
    D = 700 * norm(now[-1] - tar[-1]) # 비율 차이
    return A + B + C + D


# #### 11-4-3. MCMF Algorithm

# In[ ]:


def network_flow(now, tar, clu = 25):
    global capacity, flow, cost, adj, INF, source, sink
    v = 2 * clu + 2
    flow = [[0]*v for _ in range(v)] # 유량
    cost = [[0]*v for _ in range(v)] # 비용
    for i in range(clu):
        for j in range(clu, 2 * clu):
            cost[i][j] = get_cost(now[i], tar[j - clu])
            cost[j][i] = -cost[i][j]
        
    answer = [0, 0] # 최소 비용, 최대 유량
    while True:
        path, dist = [-1]*v, [INF]*v
        inQueue, queue = [0]*v, [source] # 다음에 방문할 정점들
        dist[source], inQueue[source] = 0, 1
        while queue:
            present = queue[0] # 현재 정점
            del queue[0]
            inQueue[present] = False
            for _next in adj[present]:
                # 최소 비용이고, 최대 유량일 경우
                if dist[_next] > dist[present] + cost[present][_next] and capacity[present][_next] - flow[present][_next] > 0:
                    dist[_next], path[_next] = dist[present] + cost[present][_next], present
                    if not inQueue[_next]:
                        queue.append(_next)
                        inQueue[_next] = 1
        if path[sink] == -1: # 가능한 모든 경로를 찾았을 경우
            break
        # 현재 경로에서의 최소 유량 찾음
        flowRate = INF
        present = sink
        while present != source:
            previous = path[present]
            flowRate = min(flowRate, capacity[previous][present] - flow[previous][present])
            present = path[present]
        # 유량 흘림
        present = sink
        while present != source:
            previous = path[present]
            answer[0] += flowRate*cost[previous][present] # 총 비용이 각 간선 비용만큼 증가
            flow[previous][present] += flowRate
            flow[present][previous] -= flowRate # 음의 유량
            present = path[present]
        answer[1] += flowRate
    return answer


# #### 11-4-4. Using linear_sum_assignment Function

# In[ ]:


def feature_row_matching(data, now, clu = 25):
    N = len(data)
    clu = len(data[0])
    fmap = np.zeros(N)

    for i in tqdm(range(N)):
        cost_table = np.zeros([clu, clu])
        for x, clu_now in enumerate(data[i]):
            for y, clu_tar in enumerate(data[now]):
                cost_table[x][y] = get_cost(clu_now, clu_tar)
        row_ind, col_ind = linear_sum_assignment(cost_table)
        answer = 0
        for p, q in zip(row_ind, col_ind):
            answer += cost_table[p][q]
        fmap[i] = answer
    return fmap


# ## 12. 선호도 기반 이미지 인덱스 샘플링

# ### 12-1. 단순 선호도 행렬 기반

# In[ ]:


def get_random_idx(scores, peak, length = 9):
    result = []
    ptg = peak ** scores
    ptg = np.cumsum(ptg)
    while len(result) != length:
        pt = random.random() * ptg[-1]
        idx = bisect.bisect_right(ptg, pt)
        if scores[idx] != 1 and idx not in result:
            result.append(idx)
    return result


# ### 12-2. 교차 선호도 행렬 기반

# In[ ]:


def get_random_idx_cross(scores, rscores, peak, length = 9):
    result = []
    ptg = peak ** (scores - rscores)
    ptg = np.cumsum(ptg)
    while len(result) != length:
        pt = random.random() * ptg[-1]
        idx = bisect.bisect_right(ptg, pt)
        if scores[idx] != 1 and idx not in result:
            result.append(idx)
    return result


# ### 12-3. Visualization of sampled index

# In[ ]:


def show_image_sampled(file_names, idx_list, plotsize = (3, 3), figsize = (12,12)):
    plt.figure(figsize = figsize);
    for i, idx in enumerate(idx_list):
        plt.subplot(plotsize[0], plotsize[1], i + 1)
        img = load_img(file_names[idx])
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.pause(0.3)    


# ## 13. 입력 결과에 따른 사용자 선호도 갱신

# ### 13-1. 단순 선호도 및 최댓값 기반

# In[ ]:


def get_new_score_linearmax(data, scores, idx_list, val_list):
    for idx, val in zip(idx_list, val_list):
        if val > 0:
            scores = np.array([max(x, y) for x, y in zip(scores, data[idx])])
    return scores


# ### 13-2. 교차 선호도 및 최댓값 기반

# In[ ]:


def get_new_score_crossmax(data, scores, rscores, idx_list, val_list):
    for idx, val in zip(idx_list, val_list):
        if val > 0:
            scores = np.array([max(x, y) for x, y in zip(scores, data[idx])])
        else:
            rscores = np.array([max(x, y) for x, y in zip(rscores, data[idx])])
    return scores, rscores


# ### 13-3. 단순 선호도 및 Softmax 함수 기반

# In[1]:


def get_new_score_softmax(data, scores, idx_list, val_list, peak):
    new_scores = np.zeros(len(scores))
    for idx, val in zip(idx_list, val_list):
        if val > 0:
            new_scores += peak ** data[idx]
        else:
            new_scores += peak ** (1 - data[idx])
    return np.array([max(x, y) for x, y in zip(scores, new_scores / (peak * len(idx_list)))])


# ### 13-4. 교차 선호도 및 binary exponential 함수 기반

# In[2]:


def get_new_score_crossexp(data, scores, rscores, idx_list, val_list, peak):
    cnt = 0
    for val in val_list:
        if not val > 0:
            cnt += 1
    new_scores = np.zeros(len(scores))
    new_rscores = np.zeros(len(scores))
    
    for idx, feature in enumerate(data[idx_list].T.copy()):
        feature = sorted(np.array([x if y > 0 else -x for x, y in zip(feature, val_list)]))
        for i in range(cnt):
            new_rscores[idx] += feature[i] * (peak ** (cnt - i - 1))
        for i in range(cnt, len(idx_list)):
            new_scores[idx] += feature[i] * (peak ** (i - cnt))

    new_rscores /= (peak ** cnt) - 1
    new_rscores *= peak - 1
    new_rscores *= -1
    new_scores /= (peak ** (len(idx_list) - cnt)) - 1
    new_scores *= peak - 1
    new_rscores = np.array([max(x, y) for x, y in zip(rscores, new_rscores)])
    new_scores = np.array([max(x, y) for x, y in zip(scores, new_scores)])
    
    return new_scores, new_rscores


# ## 14. 최종 선호도 결과에 기반한 추천

# ### 14-1. Default type

# In[ ]:


def show_image_by_score_cross(file_names, scores, rscores, length = 25, reverse = False):
    if reverse:
        new_idx = sorted(range(len(scores)), key= lambda i: scores[i] - rscores[i])[:length]
    else:
        new_idx = sorted(range(len(scores)), key= lambda i: scores[i] - rscores[i])[-length:]
        new_idx.reverse()
    print(scores[new_idx] - rscores[new_idx])
    show_image_sampled(file_names, new_idx, plotsize = (5, 5), figsize = (16, 16))
    return new_idx


# ### 14-2. Remove Duplicate

# In[ ]:


def show_image_by_score_cross_no_duplicated(file_names, scores, rscores, dup_list, length = 25, reverse = False):
    new_idx = sorted(range(len(scores)), key= lambda i: scores[i] - rscores[i])
    if not reverse:
        new_idx.reverse()
    result_idx = []
    idx = 0
    while len(result_idx) != length:
        if new_idx[idx] not in dup_list:
            result_idx.append(new_idx[idx])
        idx += 1
    print(scores[result_idx] - rscores[result_idx])
    show_image_sampled(file_names, result_idx, plotsize = (5, 5))
    return result_idx


# ### 14-3. Subfunction for check duplication

# In[ ]:


def add_duplicated_list(idx_list, val_list, dup_list):
    for idx, val in zip(idx_list, val_list):
        if val > 0:
            dup_list.append(idx)
    return dup_list


# ## 12~14. Simulating Recommendation System

# In[4]:


def recommend_images_v1(data_path, img_path):
    file_names = get_file_names(img_path, 'jpg')
    fmap = np.load(data_path)
    
    scores = np.zeros(len(fmap))
    rscores = np.zeros(len(fmap))
    dup_list = []
    
    for i in range(5):
        idx_list = get_random_idx_cross(scores, rscores, 30)
        show_image_sampled(file_names, idx_list)
        val_list = list(map(int, input().split()))
        dup_list = add_duplicated_list(idx_list, val_list, dup_list)
        scores, rscores = get_new_score_crossmax(new_fmap, scores, rscores, idx_list, val_list)
    
    result_idx = show_image_by_score_cross_no_duplicated(file_names, scores, rscores, dup_list)
    result_idx = show_image_by_score_cross_no_duplicated(file_names, scores, rscores, dup_list, reverse = True)

