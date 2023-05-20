#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import cv2


# ## Recogniser function

# In[2]:


def model(face_num, e_vector, w):
    global image_count, correct_image
    image_count += 1
    test_image = P_test[:,face_num]
    n_test_image = np.subtract(test_image,mean_matrix)

    w_test = np.dot(e_vector, n_test_image)
    d = w - w_test
    norms = np.linalg.norm(d, axis=1)
    match_img_no = np.argmin(norms)
    test_img_set = int(face_num/5)
    if (match_img_no >= (20 * test_img_set) and match_img_no < (20 * (test_img_set + 1))):
        correct_image += 1


# ## Parameters

# In[3]:


K = 5 #no of principal components


# ## Image parameters

# In[4]:


path = './data/'
file = os.listdir(path)
width = 50
height = 50
num_train = 20
num_test = 5
num_people = 5
total_train = num_train * num_people
total_test = num_test * num_people


# ## Pattern Matrix

# In[5]:


P_train = np.zeros((height*width,total_train))
for i in range(total_train):
    img = cv2.imread(path + 'train/'+str(i+1)+'.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray_img, (50, 50))
    P_train[:,i] = np.array(resize).flatten()

P_test = np.zeros((height*width,total_test))
for i in range(total_test):
    img = cv2.imread(path + 'test/'+str(i+1)+'.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray_img, (50, 50))
    P_test[:,i] = np.array(resize).flatten()


# ## Mean

# In[6]:


mean_matrix = np.mean(P_train, axis=1)


# ## Normalization

# In[7]:


n_P_train=(P_train.T-mean_matrix).T


# ## Psudo covariance 

# In[8]:


p_cov_matrix=(1/total_train)*(np.dot(n_P_train.T,n_P_train))


# ## Eigenvalues & EigenVectors

# In[9]:


eigenvalues, eigenvectors, = np.linalg.eig(p_cov_matrix)
eigen_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
eigen_pairs.sort(reverse=True)
eig_val_sort = [eigen_pairs[index][0] for index in range(len(eigenvalues))]
eig_vec_sort = [eigen_pairs[index][1] for index in range(len(eigenvalues))]


# ## Reduced eigen vector

# In[10]:


r_e_vector = np.array(eig_vec_sort[:K]).transpose()


# ## Eigen vectors of Pattern Matrix

# In[11]:


e_vector = np.dot(P_train, r_e_vector)
e_vector = e_vector.transpose()
w = np.array([np.dot(e_vector, pattern) for pattern in n_P_train.T])


# ## Testing

# In[12]:


image_count = 0
correct_image = 0
for i in range(len(P_test.T)):
    model(i, e_vector, w)

print("No of PC:", K)
print('Accuracy: {}/{} = {}%'.format(correct_image, image_count, correct_image/image_count*100.00))

