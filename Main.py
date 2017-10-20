# 'Author: Rodrigo Moreira'


import pandas as pd
import matplotlib as plt

csv_x = pd.read_csv('X.csv', sep=',', header=None)
csv_y = pd.read_csv('Y.csv', sep=',', header=None)

# Esse trecho de codigo retira a primeira linha do DataFrame (que contem os nomes das colunas), cria uma novo DataFrame sem essa primeira linha, depois adiciona as colunas na forma de indices
new_header = csv_x.iloc[0]
csv_x = csv_x[1:]
csv_x.columns = new_header

for columns in csv_x.columns:
    # print("Coluna '"+columns+"':\n%s " % csv_x[columns].astype(float).describe())
    print("Column '" + columns + "':\n")
    print("Mean: %0.f" % csv_x[columns].astype(float).mean())
    print("Maximum: %0.f" % csv_x[columns].astype(float).max())
    print("Minimum: %0.f" % csv_x[columns].astype(float).min())
    print("25 Percentil %0.2f" % csv_x[columns].astype(float).quantile(.25))
    print("90 Percentil %0.2f" % csv_x[columns].astype(float).quantile(.90))
    print("Standard Deviation: %0.2f" % csv_x[columns].astype(float).std())
    print("\n")

# -----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#

new_header = csv_y.iloc[0]
csv_y = csv_y[1:]
csv_y.columns = new_header

for columns in csv_y.columns:
    print("Column '" + columns + "':\n")
    print("Mean: %0.f" % csv_y[columns].astype(float).mean())
    print("Maximum: %0.f" % csv_y[columns].astype(float).max())
    print("Minimum: %0.f" % csv_y[columns].astype(float).min())
    print("25 Percentil %0.2f" % csv_y[columns].astype(float).quantile(.25))
    print("90 Percentil %0.2f" % csv_y[columns].astype(float).quantile(.90))
    print("Standard Deviation: %0.2f" % csv_y[columns].astype(float).std())
    print("\n")

# (a) the number of observations with memory usage larger than 80%;

print("Number of Observations with memory usage (X..memused) larger than 80%%: %.d" %
      csv_x[csv_x['X..memused'].astype(float) > 80]['X..memused'].count())

# (b) the average number of used TCP sockets for observations with more than 18.000 interrupts/sec;

print("The average number of used TCP socket for observations with more than 18.000 interrupts/sec: %0.2f" %
      csv_x[csv_x['sum_intr.s'].astype(float) > 18000]['tcpsck'].astype(float).mean())

# (c) the minimum memory utilization for observations with CPU idle time lower than 20%.

print("The minimum memory utilization for observations with CPU idle time lower than 20%%: %0.2f (X..memused)" %csv_x[csv_x['all_..idle'].astype(float) < 20]['X..memused'].astype(float).min())
