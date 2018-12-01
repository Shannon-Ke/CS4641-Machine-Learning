Shannon Ke
CS 4641
Unsupervised Learning Project

***********KMEANS************

In the python file kmeans.py, I have a portion of code that fits kmeans to my nursery dataset and
a portion of code that fits kmeans to my cars dataset. A section of code after each one is
where I actally use matplotlib to plot my data, with one always commented out. To print out
the other graph, just comment out the one originally there and uncomment the portion commented out.
Lines 53 and 98 start the portion of the code that applies PCA to k-means. Commenting those out will
yield the raw k-means algorithm, and including them will yield the k-means algorithm run on PCA formatted
data.

***********GMM**************

In the python file gmm.py, the way to toggle between running GMM on the nursery dataset vs the cars dataset
and vice versa is in lines 55-58. If you want to run GMM on the nursery dataset, keep 55-56, or else comment
those out and keep 57-58. Lines 66-70 are where I apply PCA to the GMM, which can be removed by commenting
them out. The make_ellipses prints out the graphs with respect to each covariance matrix type.

***********PCA**************

In the python file pca.py, the way to toggle between displaying one dataset vs the other involves toggling
between lines 64-65 and 66-67. To go between displaying the 3D PCA vs the 2D PCA, either 58 or 59 have to
uncommented, line 48, n_components must be 3 for 3D and 2 for 2D, and lines 62 and 72 must be uncommented for 
3D depictions.

***********NEURAL NET*************

In the python files NeuralNet_GMM.py, NeuralNet_kmeans.py, and NeuralNet_PCA.py, I have the neural networks
running with each clustering type/PCA. For the clustering files, the first commented out portion was for writing
to a new csv file the data from predicting based on each clusterign algorithm. I then had to
manually go into the csv file and move the added column to the middle so that the last column could still
be the label.