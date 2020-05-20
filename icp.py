import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from random import sample
from scipy.spatial import cKDTree
from time import time
from model_loader import *
from matrix_utils import *
import sys
import math
import cal_precision
MAXint = sys.maxsize

'''
两个点集P1，P2，每一步迭代，都朝着距离最小的目标进行。

a. 筛选点对：由P1中的点，在P2中搜索出其最近的点，组成一个点对；
找出两个点集中所有的点对。点对集合相当于进行有效计算的两个新点集。

b. 根据点集对，即两个新点集，计算两个重心。

c. 由新点集，计算出下一步计算的旋转矩阵R，和平移矩阵t(其实来源于重心的差异)。

d. 得到旋转矩阵和平移矩阵Rt，就可以计算点集P2进行刚体变换之后的新点集P2`，
由计算P2到P2`的距离平方和，以连续两次距离平方和之差绝对值，作为是否收敛的依据。
若小于阈值，就收敛，停止迭代。

e. 重复a-e，直到收敛或达到既定的迭代次数。
'''

def align(source_model, dest_model, max_it, with_error=True, Threshold=0.1):

    tree = cKDTree(np.reshape(dest_model.points, (len(dest_model.points), 3)))

    new_model = copy.deepcopy(source_model)

    it = 0
    er = 0
    errors = []

    while it < max_it:
        correspondence_list = []

        for p in new_model.points:
            '''
            p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.   
            1 is the sum-of-absolute-values "Manhattan" distance  
            2 is the usual Euclidean distance ######## 
            infinity is the maximum-coordinate-difference distance  
            遍历原始数据中的所有点，使用欧式距离查找最近邻的邻居，组成点对（步骤a）
            '''      
            d, i = tree.query(np.reshape(p, (1, 3)), k=1, p=2)
            closest = dest_model.points[i[0]]
            correspondence_list.append(closest)

        print(str(it+1) + " iterations...")
        '''
        计算下一步的旋转矩阵和平移向量（步骤c）
        '''
        r, t = calculate_translation(new_model.points, correspondence_list)
        '''
        更新模型位置（步骤d）
        '''
        new_model.apply_transformation(r, t)
        '''
        计算误差
        '''
        err = er
        er = calculate_error(new_model.points, correspondence_list, r, t)
        print(er)
        errors.append(er)
        it += 1
        if abs(err-er)<0.1:
            break

    # Plot the error of error over iterations

    if with_error:
        plt.plot(range(0, it), errors)
        plt.xlabel("Iterations")
        plt.ylabel("Magnitude of error")
        plt.show()


    return new_model,min(errors)

def subsample_align(source_model, dest_model, max_it, sample_size, with_error=True):
    tree = cKDTree(np.reshape(dest_model.points, (len(dest_model.points), 3)))

    new_model = copy.deepcopy(source_model)

    it = 0
    er = 0
    errors = []

    while it < max_it:
        correspondence_list = []
        subsample = sample(new_model.points, sample_size)

        for p in subsample:
            '''
            p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.   
            1 is the sum-of-absolute-values "Manhattan" distance  
            2 is the usual Euclidean distance ######## 
            infinity is the maximum-coordinate-difference distance  
            遍历原始数据中的所有点，使用欧式距离查找最近邻的邻居，组成点对（步骤a）
            '''      
            d, i = tree.query(np.reshape(p, (1, 3)), k=1, p=2)
            closest = dest_model.points[i[0]]
            correspondence_list.append(closest)

        print(str(it+1) + " iterations...")
        '''
        计算下一步的旋转矩阵和平移向量（步骤c）
        '''
        r, t = calculate_translation(subsample, correspondence_list)
        '''
        更新模型位置（步骤d）
        '''
        new_model.apply_transformation(r, t)
        '''
        计算误差
        '''
        err = er
        er = calculate_error(subsample, correspondence_list, r, t)
        print(er)
        errors.append(er)
        it += 1
        if abs(err-er)<0.1:
            break

    # Plot the error of error over iterations

    if with_error:
        plt.plot(range(0, it), errors)
        plt.xlabel("Iterations")
        plt.ylabel("Magnitude of error")
        plt.show()


    return new_model,min(errors)


def cal_a1(phi, omega, kappa):
    a1 = math.cos(phi)*math.cos(kappa)-math.sin(phi)*math.sin(omega)*math.sin(kappa)
    return a1
    # pass

def cal_a2(phi, omega, kappa):
    a2 = -math.cos(phi)*math.sin(kappa)-math.sin(phi)*math.sin(omega)*math.cos(kappa)
    return a2
    # pass

def cal_a3(phi, omega, kappa):
    a3 = -math.sin(phi)*math.cos(omega)
    return a3
    # pass

def cal_b1(phi, omega, kappa):
    b1 = math.cos(omega)*math.sin(kappa)
    return b1
    # pass

def cal_b2(phi, omega, kappa):
    b2 = math.cos(omega)*math.cos(kappa)
    return b2
    # pass

def cal_b3(phi, omega, kappa):
    b3 = -math.sin(omega)
    return b3
    # pass

def cal_c1(phi, omega, kappa):
    c1 = math.sin(phi)*math.cos(kappa)+math.cos(phi)*math.sin(omega)*math.sin(kappa)
    return c1
    # pass

def cal_c2(phi, omega, kappa):
    c2 = -math.sin(phi)*math.sin(kappa)+math.cos(phi)*math.sin(omega)*math.cos(kappa)
    return c2
    # pass

def cal_c3(phi, omega, kappa):
    c3 = math.cos(phi)*math.cos(omega)
    return c3
    # pass

def calculate_para(phi,omega,kappa):
    a1 = cal_a1(phi, omega, kappa)
    a2 = cal_a2(phi, omega, kappa)
    a3 = cal_a3(phi, omega, kappa)
    
    b1 = cal_b1(phi, omega, kappa)
    b2 = cal_b2(phi, omega, kappa)
    b3 = cal_b3(phi, omega, kappa)
    
    c1 = cal_c1(phi, omega, kappa)
    c2 = cal_c2(phi, omega, kappa)
    c3 = cal_c3(phi, omega, kappa)

    R = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])

    return R

def calculate_mean(points):
    mean = np.array([[0], [0], [0]])

    for i in range(0,len(points)):
        mean = mean+points[i]

    mean = mean/len(points)
    return mean

def calculate_error(p_points, q_points, rotation_matrix, translation_vector):
    error_sum = 0

    for i in range(0, len(p_points)):
        error_sum += pow(np.linalg.norm(p_points[i] - rotation_matrix.dot(q_points[i]) - translation_vector), 2)
    return error_sum/len(p_points)

def calculate_mse(p_points, q_points):
    p_pts = np.array(p_points)
    q_pts = np.array(q_points)
    xp,yp,zp = p_pts[:,0],p_pts[:,1],p_pts[:,2]
    xq,yq,zq = q_pts[:,0],q_pts[:,1],q_pts[:,2]
    deltax = cal_precision.get_mse(xp,xq)
    deltay = cal_precision.get_mse(yp,yq)
    deltaz = cal_precision.get_mse(zp,zq)

    # xp = np.array(:,0)
    return deltax**2+deltay**2+deltaz**2

#两种translation，效果一样
def calculate_translation(p_points, q_points):
    assert len(p_points) == len(q_points)
    sample_size = len(p_points)

    # Normalise to barycentric form

    normalised_points_of_p = []
    normalised_points_of_q = []
    
    '''
    根据点对计算重心（步骤b）
    '''
    mean_of_p = calculate_mean(p_points)
    mean_of_q = calculate_mean(q_points)

    #标准化
    for i in range(0, sample_size):
        normalised_points_of_p.append(p_points[i] - mean_of_p)
        normalised_points_of_q.append(q_points[i] - mean_of_q)

    # Multiply normalised barycenters together. Add to matrix.

    sum_of_products_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for i in range(0, sample_size):
        product_matrix = normalised_points_of_p[i] * (normalised_points_of_q[i].transpose())
        sum_of_products_matrix = sum_of_products_matrix + product_matrix

    # RR^T = 1!
    # svd分解
    U, S, V = np.linalg.svd(sum_of_products_matrix)
    rotation_matrix = V.transpose() @ U.transpose()
    # rotation_matrix = np.dot(V.transpose(), U.transpose())
    # t = p - Rq
    translation_vector = mean_of_p - rotation_matrix.dot(mean_of_q)
    # print(rotation_matrix, translation_vector)

    return rotation_matrix, translation_vector

def calculate_translation_(p_points, q_points):
    assert len(p_points) == len(q_points)
    sample_size = len(p_points)

    # Normalise to barycentric form

    # normalised_points_of_p = []
    # normalised_points_of_q = []
    p = []
    q = []
    
    '''
    根据点对计算重心（步骤b）
    '''
    # mean_of_p = calculate_mean(p_points)
    # mean_of_q = calculate_mean(q_points)
    mean_of_p = np.mean(p_points,axis=0)
    mean_of_q = np.mean(q_points,axis=0)

    for i in range(0, sample_size):
        normalised_points_of_p = p_points[i] - mean_of_p
        normalised_points_of_q = q_points[i] - mean_of_q
        # print(normalised_points_of_p[1][0])
        p.append([normalised_points_of_p[0][0],normalised_points_of_p[1][0],normalised_points_of_p[2][0]])
        q.append([normalised_points_of_q[0][0],normalised_points_of_q[1][0],normalised_points_of_q[2][0]])
    p = np.array(p)
    q = np.array(q)

    H = np.dot(p.T,q)
    # print(H)

    # RR^T = 1!
    # svd分解
    U, S, V = np.linalg.svd(H)
    rotation_matrix = np.dot(V.transpose(), U.transpose())
    # rotation_matrix = V.transpose() @ U.transpose()
    # t = p - Rq
    translation_vector = mean_of_p - rotation_matrix.dot(mean_of_q)
    # print(rotation_matrix, translation_vector)

    return rotation_matrix, translation_vector

def nearest_neighbor(p_points,q_points):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert len(p_points) == len(q_points)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(q_points)
    distances, indices = neigh.kneighbors(p_points, return_distance=True)
    return distances.ravel(), indices.ravel()

#普通全匹配ICP check
def run_icp():

    # m1 = ModelLoader.load("DPEX_Data11/hand-high-tri.ply")
    # m2 = ModelLoader.load("DPEX_Data11/hand-low-tri.ply")
    m1 = ModelLoader.load("output/test33_m1.ply")
    m2 = ModelLoader.load("output/test_m2.ply")  
   
    max_it = 100

    before = time()
    
    m3,er = align(m1, m2, max_it)

    ModelLoader.save("output/icp.ply", m3)
    
    after = time()

    print("ICP took " + str(after - before)+' seconds')

#加入旋转扰动 check
def run_icp_r():

    # m1 = ModelLoader.load("DPEX_Data11/hand-high-tri.ply")
    # m2 = ModelLoader.load("DPEX_Data11/hand-low-tri.ply")
    m1 = ModelLoader.load("output/test_m1.ply")
    m2 = ModelLoader.load("output/test_m1.ply")

    # # Add white noise to M2
    m2.apply_zero_mean_noise(20)
    # # Rotate M2 by some value in radians...
    perturb_deg = 20.0 * (pi / 180.0)
    perturb_R = MatrixUtils.construct_rotation_matrix(perturb_deg, np.array([0, 0, 1]))
    perturb_T = np.array([[60], [60], [60]])
    m2.apply_transformation(perturb_R,perturb_T)
   
    max_it = 60

    before = time()
    
    m3,er = align(m1, m2, max_it)

    ModelLoader.save("output/icp_dr.ply", m2)
    ModelLoader.save("output/icp_or.ply", m3)
    
    after = time()

    print("ICP took " + str(after - before)+' seconds')

#加入扰动迭代（自动检测拟合并加入扰动迭代）
def run_icp_iter():

    m1 = ModelLoader.load("output/test_m1.ply")
    m2 = ModelLoader.load("output/test_m2.ply")
    er = MAXint
    ms = copy.deepcopy(m1)
    md = copy.deepcopy(m2)
   
    max_it = 100
    iteration = True

    before = time()

    while iteration==True: 
        if er>300:
            print(er)

            perturb_degx = 30.0 * (math.pi / 180.0)
            perturb_Rx = MatrixUtils.construct_rotation_matrix(perturb_degx, np.array([1, 0, 0]))
            perturb_degy = 0.0 * (math.pi / 180.0)
            perturb_Ry = MatrixUtils.construct_rotation_matrix(perturb_degy, np.array([0, 1, 0]))
            perturb_degz = -30.0 * (math.pi / 180.0)
            perturb_Rz = MatrixUtils.construct_rotation_matrix(perturb_degz, np.array([0, 0, 1]))

            perturb_R = np.dot(np.dot(perturb_Rx,perturb_Ry),perturb_Rz)

            perturb_T = np.array([[0], [0], [0]])
            ms.apply_transformation(perturb_R,perturb_T)

            m3,er = align(ms, md, max_it, Threshold=1)

            ms = m3

            iteration = True

        elif er<300:
            ModelLoader.save("output/icp_iter.ply", m3)
            iteration = False
            break
    
    after = time()

    print("ICP took " + str(after - before)+' seconds')

#降采样ICP
def run_subsample_icp():
    # m1 = ModelLoader.load("DPEX_Data11/hand-high-tri.ply")
    # m2 = ModelLoader.load("DPEX_Data11/hand-low-tri.ply")
    m1 = ModelLoader.load("output/test33_m1.ply")
    m2 = ModelLoader.load("output/test_m2.ply") 
   
    max_it = 100
    sample_size = 4096

    before = time()
    
    m3,er = subsample_align(m1, m2, max_it, sample_size)

    ModelLoader.save("output/icp_sub.ply", m3)
    
    after = time()

    print("sub-ICP took " + str(after - before)+' seconds')

if __name__ == "__main__":
    
    # run_icp()
    # run_icp_r()
    run_icp_iter()
    # run_subsample_icp()

    pass

