#################################################################################################
#                                                                                               #
#                           Principal Components Analysis (PCA)                                 #
#                                                                                               #
# The script calculates the first two principal components (PC) of an n*d dimensional data set, # 
# transforms the dataset on the two PCs and plots the free ebergy (-log of probability) of the  #
# data-set on each of the PCs as well as on the two PCs simutaneously. Finally, it calculates   #
# the mutual information between the two PCs.                                                   #
#                                                                                               #
# Input command to run the program: python pca.py <data-file name>                              #
# Output: Three plots with the free energies and an output file named 'output.txt' with a more  #
# detailed output, like eigenvalues of the covariance matrix, eigenvectors and PC vectors       #
#                                                                                               #
# The author of the program is:                                                                 #
# Dr. Swapnil Wagle                                                                             #
# Max Planck Institute of Colloids and Interfaces                                               #
# Potsdam, Germany                                                                              #
# E-mail: swapnil.wagle@mpikg.mpg.de                                                            #
#                                                                                               #
#################################################################################################
#! /usr/env/python

import os, sys
import numpy as np
from numpy import linalg as la
import math
import matplotlib.pyplot as plt

# Command line arguments; the file path of the data-file
file_name = sys.argv[1]
out_file = open("output.txt", "w")

# Determining the number of colomns in the data set to create the numpy array of that dimension
with open(file_name) as f:
    first_line = f.readline()
    data = first_line.split()
f.close()

# Initialization of the numpy array that contains the data from the data-file and the calculated 
# averages of the data
a = np.empty([0,len(data)], dtype = float)
average = np.zeros(len(data), dtype = float)


# Creation of a 2D numpy array containing the data from the data-file
def create_vector(file_name, a):
    for lines in open(file_name):
        data = lines.split()
        if len(data) != 0:
            b = np.array(data)
            a = np.vstack([a, b.astype(float)])    
    f.close()
    return(a)
        
# For centering the data to the origin; through this process the sample-mean 
# of each of the colomn has been shifted to zero. The centered data is stored in the original
# matrix called 'a'
def center_matrix(a):
    for i in range (0,len(a[0])):
        average = np.mean(a, axis =0)
    for i in range (0,len(a[0])):
        a[:,i] = a[:,i] - average[i]
#    print("The centered matrix is")
#    print(a)
    return(a)
       
# Calculation of the covariance matrix, which is stored as a 2D numpy array
def calculate_cov_matrix(a):
    cov = np.empty([len(a[0]),len(a[0])], dtype = float)
    cov = np.mat(np.transpose(a)) * np.mat(a) 
    cov = np.true_divide(cov, len(a)-1)
    out_file.write("\nThe covariance matrix is:\n")
    out_file.write(str(cov))
    return(cov)

# Creating eigenvalue matrix and storing the eigenvalues in 'w' and the eigenvectors in 'v'
def calculate_pca(cov):
    w, v = la.eig(cov)
    out_file.write("\nThe eigenvalues and eigenvectors are:")
    out_file.write("\nEigenvalues")
    out_file.write(str(w))
    out_file.write("\nEigenvectors")
    out_file.write(str(v))
    
# Finding the two largest eigenvalues from numpy array 'w', which stores all the eigenvalues
# of the covariance matrix
    first = w[0]
    first_index = 0
    second = w[-1]
    second_index = -1
    for i in range(0, len(w)):
        if w[i] > first:
            second = first
            second_index = first_index
            first = w[i]
            first_index = i
        elif (w[i] <= first and w[i] >= second):
            second = w[i]
            second_index = i
        else:
            continue

# The 'pca_matrix' is the PCA matrix containing a 2D numpy array, the first row of which is 
# the first principal component and the second row of which is the second principal component
    pca_matrix = np.empty([2,len(a[0])], dtype = float)
    v = np.transpose(v)
    pca_matrix[0] = v[first_index]
    pca_matrix[1] = v[second_index]
    out_file.write("\nThe two principal components are:")
    out_file.write("\nFirst\n")
    out_file.write(str(pca_matrix[0]))
    out_file.write("\nSecond\n")
    out_file.write(str(pca_matrix[1]))
    return (pca_matrix)

# Transforming the origin-centered matrix of the data onto the PCA matrix
def transform_data (pca_matrix, a):
    new_matrix = np.mat(pca_matrix) * np.mat(np.transpose(a))
# Enable (uncomment) the following six lines if you want to see the scattered plot of the data 
# points on the PCs

    #fig, ax = plt.subplots()
    #ax.set_title('Data points on the first two PCs')
    #ax.set_xlabel('PCA1')
    #ax.set_ylabel('PCA2')
    #ax.scatter(new_matrix[0], new_matrix[1])
    #plt.savefig('data_on_PCs.png')
    return (new_matrix)

# Calculation of the one dimensional free energy surface (FES). The 'prob_mi' variable stores
# the probabilities for calculation of mutual information
def one_D_fes_calculation(new_matrix_array, nobins):
    hist, bins = np.histogram(new_matrix_array, bins = nobins)
    fes = np.zeros(len(hist), dtype = float)
    prob = np.empty(len(hist),dtype=float)
    prob_mi = np.empty(len(hist),dtype=float)
    grid_mid = np.empty(len(hist),dtype=float)
    summ = np.sum(hist)
    mini = np.min(hist[np.nonzero(hist)])
    mini = mini / 10
    for i in range (0,len(hist)):
        prob[i] = float(hist[i] / summ)
        prob_mi[i] = float(hist[i] / summ)
        if prob[i] <= 0:
            prob[i] = mini
        fes[i] = -1 * math.log((prob[i]), 10)
    for i in range (0,len(hist)):
        grid_mid[i] = (bins[i] + bins[i+1])/2
    return(grid_mid, fes, prob_mi)

# Calculation of the two dimensional free energy surface (FES). The 'prob2D_mi' variable stores
# the probabilities for calculation of mutual information
def two_D_fes_calculation(new_matrix, xbins, ybins):
    new_matrix_1 = np.transpose(new_matrix)
    hist2d, xedges, yedges = np.histogram2d(np.squeeze(np.asarray(new_matrix_1[:,0])), np.squeeze(np.asarray(new_matrix_1[:,1])), [xbins, ybins])
    fes2d = np.zeros([len(hist2d), len(hist2d[0])], dtype = float)
    prob2d = np.empty([len(hist2d), len(hist2d[0])],dtype=float)
    prob2d_mi = np.empty([len(hist2d), len(hist2d[0])],dtype=float)
    grid_mid2dx = np.empty(len(hist2d), dtype=float)
    grid_mid2dy = np.empty(len(hist2d[0]), dtype=float)
    summ2d = np.sum(hist2d)
    mini = np.min(hist2d[np.nonzero(hist2d)])
    mini = mini / 10
    for i in range (0,len(hist2d)):
        for j in range (0,len(hist2d[0])):
            prob2d[i][j] = float(hist2d[i][j] / summ2d)
            prob2d_mi[i][j] = float(hist2d[i][j] / summ2d)
            if prob2d[i][j] <= 0:
                prob2d[i][j] = mini
            fes2d[i][j] = -1 * math.log((prob2d[i][j]), 10)
        
    for i in range (0,len(hist2d)):
        grid_mid2dx[i] = (xedges[i] + xedges[i+1]) / 2
    for i in range (0,len(hist2d[0])):
        grid_mid2dy[i] = (yedges[i] + yedges[i+1]) / 2
    fes2d = np.transpose(fes2d)
    return(grid_mid2dx, grid_mid2dy, fes2d, prob2d_mi)

# Calculation of the mutual information
def calculate_mutual_information(prob2d_mi, prob_mi1, prob_mi2):
    temp_var = 0
    for i in range(0, len(prob2d_mi)):
        for j in range (0, len(prob2d_mi[0])):
            if (prob2d_mi[i][j] != 0 and prob_mi1[i] != 0 and prob_mi2[j] != 0):
                temp_var = temp_var + prob2d_mi[i][j] * math.log((prob2d_mi[i][j]/prob_mi1[i]/prob_mi2[j]), 10)
    return(temp_var)


a = create_vector(file_name, a)
a = center_matrix(a)
cov = calculate_cov_matrix(a)
pca_matrix = calculate_pca(cov)
new_matrix = transform_data (pca_matrix, a)

nobins1 = 30 
grid_midx, fes1, prob_mi1 = one_D_fes_calculation(new_matrix[0], nobins1)
fig, ax = plt.subplots()
ax.set_title('Plot between PCA1 and free energy on PCA1')
ax.set_xlabel('PCA1')
ax.set_ylabel('FES')
ax.plot(grid_midx, fes1)
plt.savefig('pca1_fes.png')

nobins2 = 20
grid_midy, fes2, prob_mi2 = one_D_fes_calculation(new_matrix[1], nobins2)
fig, ax = plt.subplots()
ax.set_title('Plot between PCA2 and free energy on PCA2')
ax.set_xlabel('PCA2')
ax.set_ylabel('FES')
ax.plot(grid_midy, fes2)
plt.savefig('pca2_fes.png')

xbins = 30
ybins = 20
grid_mid2dx, grid_mid2dy, fes2d, prob2d_mi = two_D_fes_calculation(new_matrix, xbins, ybins)
X, Y = np.meshgrid(grid_mid2dx, grid_mid2dy)
fig, ax = plt.subplots()
ax.set_title('Free energy plot on PCA1 and PCA2')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
cp = ax.contourf(X, Y, fes2d, 10)
fig.colorbar(cp, ax=ax)
plt.savefig('pca1_pca2_fes.png')

if ((len(prob2d_mi) == len(prob_mi1)) and (len(prob2d_mi[0]) == len(prob_mi2))):
    mi = calculate_mutual_information(prob2d_mi, prob_mi1, prob_mi2)
    out_file.write("\nthe mutual information between the first principal component and the second principal component is:")
    out_file.write(str(mi))
else:
    out_file.write("\nFor calculation of the mutual information, the grid size should be the same for the one-D FES calculation and the two-D FES calculation")
    out_file.write("Set the values of bins in the program such that xbins = nobins1 and ybins = nobins2")
