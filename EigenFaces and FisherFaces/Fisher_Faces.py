#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries
from matplotlib import pyplot as plt
from matplotlib.image import imread
from numpy.linalg import inv
import numpy as np
import os
import cv2
#Create a training and testing dataset with a proportion of 60% for training and 40% for testing.
def init():
    global img_width, img_height, num_classes, num_train_samples, train_data, i, j, img, test_data,count,num_images,acc_predictions
    data_dir_path = 'C:/Users/HP/Downloads/data 2/'
    directory = os.listdir(data_dir_path)
    img_width = 50
    img_height = 50
    num_classes = 5
    num_train_samples = 20
    number_of_testing_img = 5
    total_training_img = num_train_samples * num_classes
    total_testing_img = number_of_testing_img * num_classes 

    # to store all the training images in an array
    train_data = np.ndarray(shape=(total_training_img, img_height * img_width), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_train_samples):
            img = cv2.imread(data_dir_path + 'train1/Class' + str(i + 1) + '/' + str(j + 1) + '.png')
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            bigger = cv2.resize(gray_img,(50,50))
            train_data[num_train_samples * i + j, :] = np.array(bigger, dtype='float64').flatten()

    # to store all the testing images in an array
    test_data = np.ndarray(shape=(total_testing_img, img_height * img_width), dtype=np.float64)
    for i in range(total_testing_img):
        img = cv2.imread(data_dir_path + 'test/' + str(i + 1) + '.png')
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        bigger = cv2.resize(gray_img,(50,50))
        test_data[i, :] = np.array(bigger, dtype='float64').flatten()

    count = 0
    num_images = 0
    acc_predictions = 0

#Perform PCA on the provided data and generate a database of projected faces
def transform_using_pca(train_data, num_components):


 # Mean Matrix
    mean_face = np.mean(train_data, axis=0)

   # Normalize images
    normalized_images = train_data - mean_face

    #covariance_matrix
    covariance_matrix = np.cov(normalized_images)

    # Divide covariance matrix by number of training examples
    covariance_matrix /= train_data.shape[0]

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors, = np.linalg.eig(covariance_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

    # Feature vector 
    # Sorting the eigen pairs:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

    k_eigvectors = np.array(eigvectors_sort[:num_components]).transpose()

    #  projected data eigen space
    projected_data = np.dot(train_data.transpose(), k_eigvectors)
    projected_data = projected_data.transpose()

    # Weight matrix important features of each face
    weight_matrix = np.array([np.dot(projected_data,gray_img) for gray_img in normalized_images])

    return projected_data, weight_matrix

def fisher_faces():
    global projected_data, weight_matrix, eigvalues_sort, eigvectors_sort, k_eigvectors, FP, mean_face

    # Get the projected faces

    num_components = 18
    print("dim:", num_components)
    projected_data, weight_matrix = transform_using_pca(train_data, num_components)

    # Compute class and global mean
    class_mean = np.array([np.mean(weight_matrix[num_train_samples * i:num_train_samples * (i + 1), :], axis=0) for i in range(num_classes)])
    global_mean = np.mean(weight_matrix, axis=0)

    # Compute within and between class scatter matrices
    normalised_wc_proj_sig = weight_matrix - class_mean.repeat(num_train_samples, axis=0)
    sw = np.dot(normalised_wc_proj_sig.T, normalised_wc_proj_sig)
    normalised_proj_sig = weight_matrix - np.tile(global_mean, (num_classes * num_train_samples, 1))
    sb = np.dot(normalised_proj_sig.T, normalised_proj_sig) * num_train_samples


    # calculate the Fisherfaces
    J = np.linalg.inv(sw) @ sb
    eigenvalues, eigenvectors = np.linalg.eig(J)
    eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eigvalues_sort = [eig_pairs[i][0] for i in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[i][1] for i in range(len(eigenvalues))]


    # Compute reduced data and Fisher Projection
    num_components1 = num_classes-2
    print("k:", num_components1)
    k_eigvectors = np.array(eigvectors_sort[:num_components1]).T
    FP = np.dot(weight_matrix, k_eigvectors)
 
    # Compute mean face
    mean_face = np.mean(train_data, axis=0)


def recogniser(img_number):
    global count, highest_min, num_images, acc_predictions

    num_images += 1

    unknown_face_vector = test_data[img_number] - mean_face

    proj_fisher_test_img = np.dot(k_eigvectors.T, np.dot(projected_data, unknown_face_vector))



    # Find the index of the closest matching face
    min_norm = np.inf
    min_index = None
    for i, face in enumerate(FP):
        norm = np.linalg.norm(face - proj_fisher_test_img)
        if norm < min_norm:
            min_norm = norm
            min_index = i

    # Check if the prediction is correct
    if (min_index // 20) == (img_number // 5):
        acc_predictions += 1
    elif img_number >= 25:
        acc_predictions += 1


    count += 2
    return acc_predictions, num_images




if __name__== '__main__':
    init()
    eigvalues_sort = fisher_faces()

    # Testing all the images
    for i in range(len(test_data)):
        acc_predictions, num_images=recogniser(i)
    #accuracy
    print('Correct predictions: {}/{} = {}%'.format(acc_predictions, num_images, acc_predictions / num_images * 100.00))










# In[ ]:





# In[ ]:




