# Image-classification
EigenFaces and FisherFaces, for Image classification


EigenFaces

1. Eigenfaces, based on Principal Component Analysis (PCA), aim to represent facial images as a linear combination of a small set of eigenfaces.
2. These eigenfaces are derived from the statistical analysis of a training dataset and capture the most significant variations in facial features. 
3. By projecting an input image onto the eigenface space, it is possible to obtain a compact representation that can be used for facial recognition and classification.


FisherFaces

1. Fisherfaces, also known as Linear Discriminant Analysis (LDA), go beyond PCA by considering class separability. 
2. Rather than maximizing the variance as in PCA, Fisherfaces aim to find a projection that maximizes the ratio of between-class scatter to within-class scatter. 
3. This results in a discriminant space where classes are well-separated, enhancing the accuracy of image classification algorithms.

