from model_loader import *
import numpy as np
import math
from random import choice
import copy
from time import time
import sys
from matrix_utils import *
MAXint = sys.maxsize


def calculate_mean(points):
    mean = np.array([[0], [0], [0]])

    for i in range(0,len(points)):
        mean = mean+points[i]

    mean = mean/len(points)
    return mean

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
    print(R)

    return R

def test1():
    m1 = ModelLoader.load("DPEX_Data11/hand-high-tri.ply")
    m2 = ModelLoader.load("DPEX_Data11/hand-low-tri.ply")
    zoom = 1050
    ms = copy.deepcopy(m1)
    md = copy.deepcopy(m2)
    md.apply_zoom(zoom)
    
    mean_p = calculate_mean(ms.points)
    mean_q = calculate_mean(md.points)

    # print(mean_p-mean_q)
    R = calculate_para(0,0,0)
    # T = mean_q-mean_p   
    ms.apply_transformation(R,mean_p)         
    md.apply_transformation(R,mean_q)#md fixed    

    ModelLoader.save("output/test_m1.ply", ms)
    ModelLoader.save("output/test_m2.ply", md)

    # file.close()

def test2():
    m1 = ModelLoader.load("DPEX_Data11/hand-high-tri.ply")
    m2 = ModelLoader.load("DPEX_Data11/hand-low-tri.ply")

    pts = np.array(m1.points)
    pt = pts[:,0]

    print(pt)

def test3():
    m1 = ModelLoader.load("output/test3_m1.ply")
    # m2 = ModelLoader.load("output/test_m2.ply")

    ms = copy.deepcopy(m1)
    # md = copy.deepcopy(m2)

    perturb_deg = 120.0 * (math.pi / 180.0)
    perturb_R = MatrixUtils.construct_rotation_matrix(perturb_deg, np.array([1, 0, 0]))
    perturb_T = np.array([[0], [0], [0]])
    ms.apply_transformation(perturb_R,perturb_T)

    ModelLoader.save("output/test33_m1.ply", ms)
    # ModelLoader.save("output/test3_m2.ply", md)

if __name__ == "__main__":
    test1()

    pass
