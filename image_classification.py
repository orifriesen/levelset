import csv
from matplotlib import collections  as mc
import matplotlib.pyplot as plt
from persim import PersistenceImager
from pylab import MaxNLocator
import numpy as np
import os.path
import logging
import pandas as pd
import gudhi as gd  
from pylab import *
import PersistenceImages.persistence_images as pimg
from itertools import chain
from sklearn.cluster import KMeans
from sklearn import preprocessing
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

def get_diagrams(file, betti):
    imageRows = []
    with open(file, newline = '') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if int(row[0]) == int(betti):
                imageRows.append([float(row[1]), float(row[2])])
    return imageRows

vector_list = []

for city in ['okc', 'seattle', 'stpaul', 'pittsburgh', 'cleveland', 'indianapolis', 'chicago', 'cincinnati', 'philadelphia', 'rochester', 'hartford', 'boston', 'sacramento', 'lasvegas', 'denver', 'portland', 'jackson', 'batonrouge', 'birmingham', 'tulsa']:
    vector = []
    for race in ['white','black', 'asian', 'hispanic']:
        rel_path = 'results/ls/'+race+'/'+city+'.csv'
        print(race, city)
        file = os.path.join(os.path.dirname(__file__), rel_path)

        h0_list = get_diagrams(file, 0)
        h1_list = get_diagrams(file, 1)

        pimgr = PersistenceImager(pixel_size=1, birth_range=(0, 20), pers_range=(0, 20))
        h0_imgs = pimgr.transform(h0_list, skew=True)
        h1_imgs = pimgr.transform(h1_list, skew=True)

        imgs_conc = np.concatenate((h0_imgs, h1_imgs), axis=1)
        imgs_vec = list(chain.from_iterable(imgs_conc))
        vector.extend(imgs_vec)
    vector_list.append(imgs_vec)

# distortions = []
# for i in range(1, 10):
#     km = KMeans(
#         n_clusters=i, init='random',
#         n_init=10, max_iter=300,
#         tol=1e-04, random_state=0
#     )
#     km.fit(vector_list)
#     distortions.append(km.inertia_)
# plt.plot(range(1, 10), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()

km = KMedoids(
    n_clusters=4, init='random', max_iter=300, random_state=0
)
y_km = km.fit_predict(vector_list)
print(y_km)