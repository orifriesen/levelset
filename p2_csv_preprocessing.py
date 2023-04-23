import csv
from matplotlib import collections  as mc
import matplotlib.pyplot as pl
from pylab import MaxNLocator
import numpy as np
import os.path
import logging
import os
import re

def findPercent(rows, row, dict):
    newRow = [row[0]]
    for i in range(1, len(rows[1])):
        #newRow.append(int(row[i]) / int(rows[4][i]) * 100)
        if int(re.sub(',', '', rows[4][i])) != 0:
            newRow.append(int(re.sub(',', '', row[i])) / int(re.sub(',', '', rows[4][i])) * 100)
        else:
            newRow.append(0)
    return newRow

dict = {0:'header', 4:'TOTAL', 2:'HISPANIC%', 5:'WHITE%', 6:'BLACK%', 7:'NATIVE%', 8:'ASIAN%', 9:'PACIFIC%'}

path = os.path.join(os.path.dirname(__file__), 'p2_csvs/raw/')
newPath = os.path.join(os.path.dirname(__file__), 'p2_csvs/processed/')

for filename in os.listdir(path):
    rows = []
    file = os.path.join(path, filename)
    with open(file, newline = '') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            rows.append(row)

        newCSVrows = [rows[0]]
        for iter in [4, 2, 5, 6, 7, 8, 9]:
            rows[iter][0] = dict.get(iter)
            newCSVrows.append(findPercent(rows, rows[iter], dict))

        rows[0][0] = 'NAME'
        for i in range(len(rows[0])):
            rows[0][i] = rows[0][i].split(',')[0]

        newCSVrows[1][0] = 'percent'

        #print(newCSVrows)
        np.savetxt(os.path.join(newPath, filename), np.transpose(newCSVrows), delimiter = ',', fmt='%s')