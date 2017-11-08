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
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import random

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

plt.clf()

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

plt.clf()

#Memory Density
density = gaussian_kde(memory.astype(float))
xs = np.linspace(1,100,200)
density.covariance_factor = lambda : .5
density._compute_covariance()
plt.plot(xs,density(xs))
plt.ylabel('Prob.')
plt.xlabel('Memory Usage')
plt.show()

plt.clf()

#CPU Histogram
plt.hist(cpu.astype(float), 100, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.03])
plt.xlabel('% CPU Usage')
plt.show()

plt.clf()

#Memory Histogram
plt.hist(memory.astype(float), 50, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.10])
plt.xlabel('% Memory Usage')
plt.show()

plt.clf()

#CPU Box Plot
plt.boxplot(cpu.astype(float), 1)
plt.xlabel('% CPU Usage')
plt.axis([0, 3, 0, 100])
plt.show()

plt.clf()

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



csv_x = csv_x.iloc[:, csv_x.columns != "TimeStamp"]
csv_y = csv_y.iloc[:, csv_y.columns != "TimeStamp"]

#x_train = csv_x.iloc[:-30, csv_x.columns != "TimeStamp"]
#x_test = csv_x.iloc[-30:, csv_x.columns != "TimeStamp"]
#y_train = csv_y.DispFrames[:-30]
#y_test = csv_y.DispFrames[-30:]


csv_x = pd.DataFrame(csv_x)
csv_y = pd.DataFrame(csv_y)

x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30, random_state=1)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

#---------A--------------

# The coefficients
print("Coefficients: ")
print(regr.coef_)

#---------B--------------
# The Normalized mean Absolute Error
print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred))

print("The Accuracy Score: %0.2f " % regr.score(x_test, y_test))

plt.clf()

#---------C--------------

plt.scatter(y_pred, y_test, color='black', alpha=0.5, marker='o')
plt.xlabel("Tests")
plt.ylabel("Video on Demand Frame Rate")
plt.show()


plt.clf()


#Setting y_test in numpy array format
y_test = np.array(y_test[0])

#---------D--------------

#Y Test Set Density
density = gaussian_kde(y_test)
density.covariance_factor = lambda : .25
density._compute_covariance()
xs = np.linspace(1,100,80, endpoint=True)
plt.plot(xs,density(xs))
plt.xlabel('Video Frame Rate')

#Y Test Set Histogram
plt.hist(y_test.astype(float), bins=1, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.1])
plt.xlabel('% Y Test Set')
plt.show()

plt.clf()

#---------E--------------

#print(y_test)
y_pred = pd.DataFrame(y_pred)
y_pred = np.array(y_pred[0])
#print(y_pred)

prediction_erros = y_test - y_pred
#print(prediction_erros)

#Prediction Erros Density
density = gaussian_kde(prediction_erros)
density.covariance_factor = lambda : .25
density._compute_covariance()
xs = np.linspace(-10, 100, 80, endpoint=True)
plt.plot(xs,density(xs))
plt.xlabel('Predicion Erros Density')
plt.show()

#---------Part 2 - Task II ---------------------

#---------A--------------


#Original
# x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30, random_state=1)

#For 50 times

x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, train_size=0.30, random_state=1)

# Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

nmae50 = np.array([])
nmae100 = np.array([])
nmae500 = np.array([])
nmae1000 = np.array([])
nmae2520 = np.array([])

for _ in range(50):

    x = pd.DataFrame(x_train)
    x_train50 = x.loc[random.sample(list(x.index), 50)]
    x_train100 = x.loc[random.sample(list(x.index), 100)]
    x_train500 = x.loc[random.sample(list(x.index), 500)]
    x_train1000 = x.loc[random.sample(list(x.index), 1000)]

    y = pd.DataFrame(y_train)
    y_train50 = y.loc[random.sample(list(x.index), 50)]
    y_train100 = y.loc[random.sample(list(x.index), 100)]
    y_train500 = y.loc[random.sample(list(x.index), 500)]
    y_train1000 = y.loc[random.sample(list(x.index), 1000)]

    regr = linear_model.LinearRegression()
    regr.fit(x_train50, y_train50)
    y_pred50 = regr.predict(x_test)

    nmae50 = np.append(nmae50, mean_absolute_error(y_test, y_pred50))
    #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred50))



    regr = linear_model.LinearRegression()
    regr.fit(x_train100, y_train100)
    y_pred100 = regr.predict(x_test)

    nmae100 = np.append(nmae100, mean_absolute_error(y_test, y_pred100))
    #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred100))

    regr = linear_model.LinearRegression()
    regr.fit(x_train500, y_train500)
    y_pred500 = regr.predict(x_test)

    nmae500 = np.append(nmae500, mean_absolute_error(y_test, y_pred500))

    #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred500))

    regr = linear_model.LinearRegression()
    regr.fit(x_train1000, y_train1000)
    y_pred1000 = regr.predict(x_test)

    nmae1000 = np.append(nmae1000, mean_absolute_error(y_test, y_pred1000))

    #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred1000))

print(nmae50)
print(nmae100)
print(nmae500)
print(nmae1000)

#NMAE for Samples size of 50
plt.boxplot(nmae50, 1)
plt.xlabel('N. M. A. E. 50')
plt.axis([0, 1.5, 0, 100])
plt.show()

#     #Seto novamente a configuracao de DataFrame para nao perder a dimensao
#     x_train50 = pd.DataFrame(x_train50)
#     x_test50 = pd.DataFrame(x_test50)
#     y_train50 = pd.DataFrame(y_train50)
#     y_test50 = pd.DataFrame(y_test50)
#
#     regr = linear_model.LinearRegression()
#     regr.fit(x_train50, y_train50)
#     y_pred50 = regr.predict(x_test50)
#
#     # The Normalized mean Absolute Error
#     print(mean_absolute_error(y_test50, y_pred50))
#     nmae50 = np.append(nmae50, mean_absolute_error(y_test50, y_pred50))
#
# print(nmae50)
#
# #For 100 times
# x_train100, x_test100, y_train100, y_test100 = train_test_split(x_train, y_train, train_size=100, random_state=1)
#
# #Seto novamente a configuracao de DataFrame para nao perder a dimensao
# x_train100 = pd.DataFrame(x_train100)
# x_test100 = pd.DataFrame(x_test100)
# y_train100 = pd.DataFrame(y_train100)
# y_test100 = pd.DataFrame(y_test100)
#
# #For 200 times
# x_train200, x_test200, y_train200, y_test200 = train_test_split(x_train, y_train, train_size=200, random_state=1)
#
# #Seto novamente a configuracao de DataFrame para nao perder a dimensao
# x_train200 = pd.DataFrame(x_train200)
# x_test200 = pd.DataFrame(x_test200)
# y_train200 = pd.DataFrame(y_train200)
# y_test200 = pd.DataFrame(y_test200)
#
# #For 500 times
# x_train500, x_test500, y_train500, y_test500 = train_test_split(x_train, y_train, train_size=500, random_state=1)
#
# #Seto novamente a configuracao de DataFrame para nao perder a dimensao
# x_train500 = pd.DataFrame(x_train500)
# x_test500 = pd.DataFrame(x_test500)
# y_train500 = pd.DataFrame(y_train500)
# y_test500 = pd.DataFrame(y_test500)
#
# #For 1000 times
# x_train1000, x_test1000, y_train1000, y_test1000 = train_test_split(x_train, y_train, train_size=1000, random_state=1)
#
# #Seto novamente a configuracao de DataFrame para nao perder a dimensao
# x_train1000 = pd.DataFrame(x_train1000)
# x_test1000 = pd.DataFrame(x_test1000)
# y_train1000 = pd.DataFrame(y_train1000)
# y_test1000 = pd.DataFrame(y_test1000)