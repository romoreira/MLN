# 'Author: Rodrigo Moreira'


import time
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from scipy.stats import gaussian_kde

csv_x = pd.read_csv('X.csv', sep=',', header=None)
csv_y = pd.read_csv('Y.csv', sep=',', header=None)

#-----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#
#TASK II

# Esse trecho de codigo retira a primeira linha do DataFrame (que contem os nomes das colunas), cria uma novo DataFrame sem essa primeira linha,
#depois adiciona as colunas na forma de indices
new_header = csv_x.iloc[0]
csv_x = csv_x[1:]
csv_x.columns = new_header

new_header = csv_y.iloc[0]
csv_y = csv_y[1:]
csv_y.columns = new_header

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

#-----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#

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

print("Number of Observations with memory usage (X_memused) larger than 80%%: %.d" %
      csv_x[csv_x['X_memused'].astype(float) > 80]['X_memused'].count())

# (b) the average number of used TCP sockets for observations with more than 18.000 interrupts/sec;

print("The average number of used TCP socket for observations with more than 18.000 interrupts/sec: %0.2f" %
      csv_x[csv_x['sum_intr.s'].astype(float) > 18000]['tcpsck'].astype(float).mean())

# (c) the minimum memory utilization for observations with CPU idle time lower than 20%.

print("The minimum memory utilization for observations with CPU idle time lower than 20%%: %0.2f (X_memused)" %csv_x[csv_x['all_idle'].astype(float) < 20]['X_memused'].astype(float).min())


# Time series of percentage of idle CPU and used memory (in same plot)
cpu = np.array(csv_x['all_idle'])
memory = np.array(csv_x['X_memused'])

plt.plot(cpu)
plt.plot(memory)
plt.ylabel('CPU x Memory')
plt.show()

# Density plots, histograms, and box plots of idle CPU and used memory

#CPU Density
density = gaussian_kde(cpu.astype(float))
xs = np.linspace(1,100,200)
density.covariance_factor = lambda : .5
density._compute_covariance()
plt.plot(xs,density(xs))
plt.ylabel('Prob.')
plt.xlabel('CPU Usage')
plt.show()

#Memory Density
density = gaussian_kde(memory.astype(float))
xs = np.linspace(1,100,200)
density.covariance_factor = lambda : .5
density._compute_covariance()
plt.plot(xs,density(xs))
plt.ylabel('Prob.')
plt.xlabel('Memory Usage')
plt.show()

#CPU Histogram
plt.hist(cpu.astype(float), 100, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.03])
plt.xlabel('% CPU Usage')
plt.show()

#Memory Histogram
plt.hist(memory.astype(float), 50, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.10])
plt.xlabel('% Memory Usage')
plt.show()

#CPU Box Plot
plt.boxplot(cpu.astype(float), 1)
plt.xlabel('% CPU Usage')
plt.axis([0, 3, 0, 100])
plt.show()

#Memory Box Plot
plt.boxplot(memory.astype(float), 1)
plt.xlabel('% Memory Usage')
plt.axis([0, 3, 0, 100])
plt.show()

#-----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#
#TASK II

#Calculando a correlacao de ("X_memused") entre as colunas para escolher qual atributo mais impacta na
#qualidade de servico de entrega de video.


for columns in csv_x.columns:
    csv_x[columns] = csv_x[columns].convert_objects(convert_numeric=True)#Transformando os dados do DataFrameX para tipo Numerico

for columns in csv_y.columns:
    csv_y[columns] = csv_y[columns].convert_objects(convert_numeric=True)#Transformando os dados do DataFrameY para tipo Numerico

#Check if I am really ignoring the first column (TimeStamp) on calc of the correlation;
#print("\nCorrelation between 'X_memused' and another columns are:  \n", csv_x.corr()['X_memused']['all_idle':])

x_train = csv_x.iloc[:-20, csv_x.columns != "TimeStamp"]
x_test = csv_x.iloc[-20:, csv_x.columns != "TimeStamp"]
y_train = csv_y.DispFrames[:-20]
y_test = csv_y.DispFrames[-20:]


#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# The coefficients
print("Coefficients: ")
print(regr.coef_)

# The Normalized mean Absolute Error
print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred))

#print(y_pred)
#print(y_test)

# Plot outputs
plt.scatter(x_test['tcpsck'], y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()