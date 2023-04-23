import csv
from matplotlib import collections  as mc
import matplotlib.pyplot as pl
from pylab import MaxNLocator
import numpy as np
import os.path
import logging
import pandas as pd
import gudhi as gd  
from pylab import *
import PersistenceImages.persistence_images as pimg

def get_barcodes(file, betti):
    rows = []
    imageRows = []
    with open(file, newline = '') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            rows.append([int(row[0]), (float(row[1]), float(row[2]))])
            imageRows.append([float(row[1]), float(row[2])])
    return rows, imageRows

for city in ['okc', 'seattle', 'stpaul', 'pittsburgh', 'cleveland', 'indianapolis', 'chicago', 'cincinnati', 'philadelphia', 'rochester', 'hartford', 'boston', 'sacramento', 'lasvegas', 'denver', 'portland', 'jackson', 'batonrouge', 'birmingham', 'tulsa']:
    for race in ['white','black', 'asian', 'hispanic']:
        rel_path = 'results/ls/'+race+'/'+city+'.csv'
        print(race, city)
        file = os.path.join(os.path.dirname(__file__), rel_path)
        barcode, image_list = get_barcodes(file)

        ax = gd.plot_persistence_diagram(barcode, legend=True)
        ax.set_aspect("equal")
        ax.set_title("Persistence diagram of " + race + " " + city)
        out_path = 'results/diagrams/ls/'+race+'/'+city+'.png'
        plt.savefig(os.path.join(os.path.dirname(__file__), out_path))

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