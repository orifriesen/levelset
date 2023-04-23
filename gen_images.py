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

def get_diagrams(file, betti):
    imageRows = []
    with open(file, newline = '') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if int(row[0]) == int(betti):
                imageRows.append([float(row[1]), float(row[2])])
    return imageRows

for city in ['okc']:
    for race in ['white','black', 'asian', 'hispanic']:
        for betti in ['0', '1']:
            rel_path = 'results/ls/'+race+'/'+city+'.csv'
            print(race, city)
            file = os.path.join(os.path.dirname(__file__), rel_path)
            image_list = get_diagrams(file, betti)

            pimgr = PersistenceImager(pixel_size=1, birth_range=(0, 20), pers_range=(0, 20))
            pdgms = image_list
            pimgs = pimgr.transform(pdgms, skew=True)

            fig, axs = plt.subplots(1, 1, figsize=(5,5))
            plt.subplots
            axs.set_title("Persistence image for " + race + " " + city + " " + betti)
            pimgr.plot_image(pimgs, ax=axs)

            plt.tight_layout()
            rel_path = 'results/images/ls/'+race+'/'+city+'_'+betti+'_pi.png'
            fig.savefig(os.path.join(os.path.dirname(__file__), rel_path))

# for city in ['okc', 'seattle', 'stpaul', 'pittsburgh', 'cleveland', 'indianapolis', 'chicago', 'cincinnati', 'philadelphia', 'rochester', 'hartford', 'boston', 'sacramento', 'lasvegas', 'denver', 'portland', 'jackson', 'batonrouge', 'birmingham', 'tulsa']:
#     barcode = []
#     for race in ['white','black', 'asian', 'hispanic']:
#         rel_path = 'results/ls/'+race+'/'+city+'.csv'
#         file = os.path.join(os.path.dirname(__file__), rel_path)
#         barcode.extend(get_barcodes(file))
#     ax = gd.plot_persistence_diagram(barcode, legend=True)
#     ax.set_aspect("equal")
#     ax.set_title("Persistence diagram of " + city)

#     out_path = 'results/diagrams/ls/total/'+city+'.png'
#     plt.savefig(os.path.join(os.path.dirname(__file__), out_path))