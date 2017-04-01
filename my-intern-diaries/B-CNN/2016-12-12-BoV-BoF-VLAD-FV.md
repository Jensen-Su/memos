---
Image vector representation
---
    2016-12-12, Monday, Cloudy
    By Jincheng Su @ Hikvision, mail: jcsu14@fudan.edu.cn


#### Bag-of-features (BoF)

Related papers:
    [1] J. Sivic and A. Zisserman. Video Google: A text retrieval approach to object matching in videos. In ICCV, pages 1470–1477, 2003
    [2] H. Jegou, M.Douze, and C.schmid. Packing bag-of-features. In ICCV, Septemtber 2009.

Main Steps:
    1. Obtain a codebook of $k$ "visual words" usually by $k$-means clustering.
    2. Assign each local descriptor from an image (new) to the closest centroid.
    3. Obtain a $k$-dimensional vector as the image representation from the histogram of the assignment of all image descriptors to visual words.
    4. Normalization. (Manhattan distance/Euclidean Normalization/$l_2$ Normalization).
    5. The final vector representation is obtained by weighting the vector components by $idf$ (inverse document frequency) terms.

#### Bag-of-visual-words (BoVW)

Related papers:
    [1] Visual Categorization with Bags of Keypoints, ECCV 2004

    1. Detection and description of image patches.
    2. Assigning patch descriptors to a set of predetermined clusters (a vocabulary) with a vector quantization algorithm.
    3. Constructing a *bag of keypoints*, which counts the number of patches assigned to each cluster.
    4. Applying a multi-class classifier, treating the bag of keypoints as the feature vector, and thus determine which category or categories to assign to the image.


#### Fisher Kernel
Related papers:
    [1] *Fisher kernels on visual vocabularies for image categorization*. In CVPR 2007.
    [2] Perronnin F, Sánchez J, Mensink T. *Improving the Fisher Kernel for Large-Scale Image Classification*[C]. Computer Vision - ECCV 2010, September 5-11, 2010, Proceedings. 2010:119-133.
    [3] Nchez J, Perronnin F, Mensink T, et al. *Image Classification with the Fisher Vector: Theory and Practice*[J]. International Journal of Computer Vision, 2013, 105(3):222-245.

The description vector is the gradient of the sample's likelihood with respect to the parameters of this distribution, scaled by the inverse square root of the **Fisher information matrix**.'

[1] models the visual words with a Gaussian mixture model (GMM), restricted to diagonal variance matrices for each of the $k$ components of the mixture. Deriving a diagonal approximation of the Fisher matrix of a GMM, they obtain a $(2d+1)*k - 1$ dimensional vector representation of an image feature set.


#### Vector Of Locally Aggregated Descriptor (VLAD)
Related papers:
    [1]. Jegou H, Douze M, Schmid C, et al. Aggregating local descriptors into a compact image representation[J]. 2010, 238(6):3304-3311.
    [2]. Arandjelovic R, Zisserman A. All About VLAD[J]. 2013, 9(4):1578-1585.
    [3]. Arandjelović R, Gronat P, Torii A, et al. NetVLAD: CNN architecture for weakly supervised place recognition[J]. Computer Science, 2016.
    [4]. Babenko A, Lempitsky V. Aggregating Deep Convolutional Features for Image Retrieval[J]. Computer Science, 2015.

#### Understand image representations

Related papers:
    Mahendran, Aravindh, and A. Vedaldi. "Understanding deep image representations by inverting them." Computer Science (2014):5188-5196.

