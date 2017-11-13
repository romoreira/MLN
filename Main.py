# 'Author: Rodrigo Moreira'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import gaussian_kde
from sklearn.cross_validation import train_test_split

def time_series_plot(teste, predito, title):

    interval = np.arange(0, 1080, 1)

    plt.plot(interval, teste, 'blue', interval, predito, 'red')
    plt.title('Time series plot with measurements and model estimations - '+title)
    plt.ylabel('Frames/sec')
    plt.show()


def scatter_plot(y_test, y_predict, y_predict_naive):
    #print("Y_test")
    #print(y_test)
    y_predict = pd.DataFrame(y_predict)
    #print(y_predict)

    #print("Y pred naive")
    #print(y_pred_naive)

    y_pred_naive['TimeStamp'] = y_test['TimeStamp']
    y_predict['TimeStamp'] = y_test['TimeStamp']

    #print(y_predict)
    #print(y_predict[0])

    a = plt.scatter(y_pred_naive.iloc[:, y_pred_naive.columns != 0], y_pred_naive.iloc[:, y_pred_naive.columns != "TimeStamp"], c="b", marker=".")
    b = plt.scatter(y_predict.iloc[:, y_predict.columns != 0], y_predict.iloc[:, y_predict.columns != "TimeStamp"], c="r", marker="o")
    c = plt.scatter(y_test.iloc[:, y_test.columns != "DispFrames"], y_test.iloc[:, y_test.columns != "TimeStamp"], c="g",  marker=">")

    plt.legend((a, b, c), ('Naive-based Predictions', 'Default Predictions', 'Target Values'), scatterpoints=1, loc='lower left', ncol=3, fontsize=10.8)


    plt.show()

def naive_method_predict(x_train_naive, y_train_naive, x_test_naive):
    x_train_naive = pd.DataFrame(x_train_naive, columns=['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1', 'tcpsck', 'pgfree.s'])
    x_test_naive = pd.DataFrame(x_test_naive, columns=['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1', 'tcpsck', 'pgfree.s'])
    y_train_naive = pd.DataFrame(y_train_naive, columns=['TimeStamp','DispFrames'])
    mean = y_train_naive.iloc[:, y_train_naive.columns != "TimeStamp"].astype(float).mean()
    mean = mean[0]


    #print(x_train_naive)
    #print(len(y_train_naive))
    #print("mean: ")
    #print(mean)

    #print(y_train_naive)

    i = 0
    for i in range(len(y_train_naive)):
        y_train_naive.iloc[i][1] = mean

    #print("Depois")
    #print(y_train_naive)
    #print("X teste naive")
    #print(x_test_naive)


    regr = linear_model.LinearRegression()
    regr.fit(x_train_naive.iloc[:, x_train_naive.columns != "TimeStamp"], y_train_naive.iloc[:, y_train_naive.columns != "TimeStamp"])
    y_pred_naive = regr.predict(x_test_naive.iloc[:, x_test_naive.columns != "TimeStamp"])
    #print("Y predito")
    #print(y_pred_naive)
    return y_pred_naive

def nmae(y_real, y_predito):

    #print("Antes")
    #print(y_real)
    #print(y_predito)

    #Set of the DataFrame configs
    y_real = pd.DataFrame(y_real)
    y_predito = pd.DataFrame(y_predito)

    #print("Depois")
    #print(y_real.iloc[:, y_real.columns != "TimeStamp"])
    #print("Media: %.2f" % y_real.iloc[:, y_real.columns != "TimeStamp"].astype(float).mean())
    #print(y_predito)

    #print("Y real: ")
    #print(y_real.iloc[0][1])

    #Loop variables initializing
    somatorio = 0.0
    m = 0
    for m in range(len(y_real)):
        somatorio += abs((y_real.iloc[m][1] - y_predito.iloc[m]))
        m += 1


    #Ajustes
    media = y_real.iloc[:, y_real.columns != "TimeStamp"].astype(float).mean()
    media = media[0]
    somatorio = somatorio[0]


    #N. M. A. E. Accuracy Measures
    nmae_resultado = (somatorio/m)/media

    return nmae_resultado

csv_x = pd.read_csv('X.csv', sep=',', header=None)
csv_y = pd.read_csv('Y.csv', sep=',', header=None)

#-----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#
#TASK I

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
    print("Mean: %0.2f" % csv_x[columns].astype(float).mean())
    print("Maximum: %0.f" % csv_x[columns].astype(float).max())
    print("Minimum: %0.f" % csv_x[columns].astype(float).min())
    print("25 Percentil %0.2f" % csv_x[columns].astype(float).quantile(.25))
    print("90 Percentil %0.2f" % csv_x[columns].astype(float).quantile(.90))
    print("Standard Deviation: %0.2f" % csv_x[columns].astype(float).std())
    print("\n")

#-----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#

for columns in csv_y.columns:
    print("Column '" + columns + "':\n")
    print("Mean: %0.2f" % csv_y[columns].astype(float).mean())
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

plt.plot(cpu, label='CPU')
plt.plot(memory, label='Memory')
legend = plt.legend(loc='lower left', shadow=True)
plt.title("CPU and Memory")
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
#plt.ylabel('Prob.')
plt.xlabel('CPU Usage')
plt.title("Density")
plt.show()

plt.clf()

#Memory Density
density = gaussian_kde(memory.astype(float))
xs = np.linspace(1,100,200)
density.covariance_factor = lambda : .5
density._compute_covariance()
plt.plot(xs,density(xs))
#plt.ylabel('Prob.')
plt.xlabel('Memory Usage')
plt.title("Density")
plt.show()

plt.clf()

#CPU Histogram
plt.hist(cpu.astype(float), 100, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.03])
plt.xlabel('% CPU Usage')
plt.title("Histogram")
plt.show()

plt.clf()

#Memory Histogram
plt.hist(memory.astype(float), 50, normed=1, facecolor='green', alpha=0.75)
plt.axis([0, 100, 0, 0.10])
plt.xlabel('% Memory Usage')
plt.title("Histogram")
plt.show()

plt.clf()

#BoxPlot CPU and Memory
cpu = np.array(cpu.astype(float))
memory = np.array(memory.astype(float))
print(cpu)
data_to_plot = [cpu, memory]
plt.boxplot(data_to_plot)
plt.xlabel('Box Plot of Idle CPU and Memory')
plt.axis([0, 3, -5, 100])
plt.xticks([1, 2], ['CPU', 'Memory'])
plt.title("Box Plot")
plt.show()



# #CPU Box Plot
# plt.boxplot(cpu.astype(float), 1)
# plt.xlabel('% CPU Usage')
# plt.axis([0, 3, 0, 100])
# plt.show()
#
# plt.clf()
#
# #Memory Box Plot
# plt.boxplot(memory.astype(float), 1)
# plt.xlabel('% Memory Usage')
# plt.axis([0, 3, -5, 100])
# plt.show()

#-----------------#---------------#----------------#------------------#-----------------#--------------------#-----------------#--------------#
#TASK II

for columns in csv_x.columns:
    csv_x[columns] = csv_x[columns].convert_objects(convert_numeric=True)#Transformando os dados do DataFrameX para tipo Numerico

for columns in csv_y.columns:
    csv_y[columns] = csv_y[columns].convert_objects(convert_numeric=True)#Transformando os dados do DataFrameY para tipo Numerico


#Retira o TimeStamp
#csv_x = csv_x.iloc[:, csv_x.columns != "TimeStamp"]
#csv_y = csv_y.iloc[:, csv_y.columns != "TimeStamp"]


#Regression Method
csv_x = pd.DataFrame(csv_x)
csv_y = pd.DataFrame(csv_y)



x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train, columns=['TimeStamp','all_idle','X_memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
x_test = pd.DataFrame(x_test, columns=['TimeStamp','all_idle','X_memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
y_train = pd.DataFrame(y_train, columns=['TimeStamp','DispFrames'])
y_test = pd.DataFrame(y_test, columns=['TimeStamp','DispFrames'])

#Backup do y_test - Utilizado na rodagem dos 50 experimentos com os 6 tamanhos da amostra
y_test_bkp = y_test
x_test_bkp = x_test

regr = linear_model.LinearRegression()
regr.fit(x_train.iloc[:, x_train.columns != "TimeStamp"], y_train.iloc[:, y_train.columns != "TimeStamp"])
y_pred = regr.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])



#---------A--------------

# The coefficients
print("Coefficients: ")
print(regr.coef_)

#---------B--------------

# The Normalized mean Absolute Error
#Regression Method
print("The Normalized Mean Absolute Error (Regression Method): %0.2f " % nmae(y_test, y_pred))

#Naive Method
csv_x = pd.DataFrame(csv_x)
csv_y = pd.DataFrame(csv_y)
x_train_naive, x_test_naive, y_train_naive, y_test_naive = train_test_split(csv_x, csv_y, test_size=0.30)
y_test_naive = pd.DataFrame(y_test_naive, columns=['TimeStamp','DispFrames'])
y_pred_naive = naive_method_predict(x_train_naive, y_train_naive, x_test_naive)


print("The Normalized Mean Absolute Error (Naive Prediction): %0.2f " % nmae(y_test_naive, y_pred_naive))

#---------C--------------
#Scatter Plot - Regression Method and Regression (Naive-based) Method
#Some conversions
y_test_naive = pd.DataFrame(y_test_naive)
y_pred_naive = pd.DataFrame(y_pred_naive)
scatter_plot(y_test, y_pred, y_pred_naive)



#---------D--------------
#Y Test Set Density
y_test = np.array(y_test["DispFrames"])
density = gaussian_kde(y_test)
density.covariance_factor = lambda : .25
density._compute_covariance()
xs = np.linspace(1,100,80, endpoint=True)
plt.plot(xs,density(xs))
plt.xlabel('Video Frame Rate')

#Y Test Set Histogram
plt.hist(y_test.astype(float), bins=1, normed=1, facecolor='green', alpha=0.75)
plt.axis([-30, 100, 0, 0.3])
plt.title("Density and Histogram Plot - Video Frame Rate Values")
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
xs = np.linspace(-30, 100, 80, endpoint=True)
plt.plot(xs,density(xs))
plt.xlabel('Predicion Erros Density')
plt.show()

#---------Part 2 - Task II ---------------------

csv_x_2 = pd.read_csv('X_2.csv', sep=',', header=None)
csv_y_2 = pd.read_csv('Y_2.csv', sep=',', header=None)

# Esse trecho de codigo retira a primeira linha do DataFrame (que contem os nomes das colunas), cria uma novo DataFrame sem essa primeira linha,
#depois adiciona as colunas na forma de indices
new_header = csv_x_2.iloc[0]
csv_x_2 = csv_x_2[1:]
csv_x_2.columns = new_header

new_header = csv_y_2.iloc[0]
csv_y_2 = csv_y_2[1:]
csv_y_2.columns = new_header

#print(csv_x_2)
#print(csv_y_2)

for columns in csv_x_2.columns:
    csv_x_2[columns] = csv_x_2[columns].convert_objects(convert_numeric=True)#Transformando os dados do DataFrameX para tipo Numerico

for columns in csv_y_2.columns:
    csv_y_2[columns] = csv_y_2[columns].convert_objects(convert_numeric=True)#Transformando os dados do DataFrameY para tipo Numerico


csv_x_2 = pd.DataFrame(csv_x_2)
csv_y_2 = pd.DataFrame(csv_y_2)

x_train, x_test, y_train, y_test = train_test_split(csv_x_2, csv_y_2, test_size=0.30)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train, columns=['TimeStamp','all_idle','X_memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
x_test = pd.DataFrame(x_test, columns=['TimeStamp','all_idle','X_memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
y_train = pd.DataFrame(y_train, columns=['TimeStamp','DispFrames'])
y_test = pd.DataFrame(y_test, columns=['TimeStamp','DispFrames'])



#---------A--------------

#For 50 times

nmae50 = np.array([])
nmae100 = np.array([])
nmae200 = np.array([])
nmae500 = np.array([])
nmae1000 = np.array([])
nmae2520 = np.array([])

for _ in range(50):

    #---To 50 samples------
    x_train50, x_test50, y_train50, y_test50 = train_test_split(x_train, y_train, train_size=50, test_size=50)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao - 50
    x_train50 = pd.DataFrame(x_train50, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    x_test50 = pd.DataFrame(x_test50, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    y_train50 = pd.DataFrame(y_train50, columns = ['TimeStamp', 'DispFrames'])
    y_test50 = pd.DataFrame(y_test50, columns = ['TimeStamp', 'DispFrames'])

    regr = linear_model.LinearRegression()
    regr.fit(x_train50.iloc[:, x_train50.columns != "TimeStamp"], y_train50.iloc[:, y_train50.columns != "TimeStamp"])
    y_pred50 = regr.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])

    nmae50 = np.append(nmae50, nmae(y_test, y_pred50))


    # ---To 100 samples------
    x_train100, x_test100, y_train100, y_test100 = train_test_split(x_train, y_train, train_size=100, test_size=100)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao - 100
    x_train100 = pd.DataFrame(x_train100, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    x_test100 = pd.DataFrame(x_test100, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    y_train100 = pd.DataFrame(y_train100, columns = ['TimeStamp', 'DispFrames'])
    y_test100 = pd.DataFrame(y_test100, columns = ['TimeStamp', 'DispFrames'])

    regr = linear_model.LinearRegression()
    regr.fit(x_train100.iloc[:, x_train100.columns != "TimeStamp"], y_train100.iloc[:, y_train100.columns != "TimeStamp"])
    y_pred100 = regr.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])

    nmae100 = np.append(nmae100, nmae(y_test, y_pred100))

    # ---To 200 samples------
    x_train200, x_test200, y_train200, y_test200 = train_test_split(x_train, y_train, train_size=200, test_size=200)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao - 200
    x_train200 = pd.DataFrame(x_train200, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    x_test200 = pd.DataFrame(x_test200, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    y_train200 = pd.DataFrame(y_train200, columns = ['TimeStamp', 'DispFrames'])
    y_test200 = pd.DataFrame(y_test200, columns = ['TimeStamp', 'DispFrames'])

    regr = linear_model.LinearRegression()
    regr.fit(x_train200.iloc[:, x_train200.columns != "TimeStamp"], y_train200.iloc[:, y_train200.columns != "TimeStamp"])
    y_pred200 = regr.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])

    nmae200 = np.append(nmae100, nmae(y_test, y_pred200))

    # ---To 500 samples------
    x_train500, x_test500, y_train500, y_test500 = train_test_split(x_train, y_train, train_size=500, test_size=500)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao - 500
    x_train500 = pd.DataFrame(x_train500, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    x_test500 = pd.DataFrame(x_test500, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    y_train500 = pd.DataFrame(y_train500, columns = ['TimeStamp', 'DispFrames'])
    y_test500 = pd.DataFrame(y_test500, columns = ['TimeStamp', 'DispFrames'])

    regr = linear_model.LinearRegression()
    regr.fit(x_train500.iloc[:, x_train500.columns != "TimeStamp"], y_train500.iloc[:, y_train500.columns != "TimeStamp"])
    y_pred500 = regr.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])

    nmae500 = np.append(nmae500, nmae(y_test, y_pred500))

    # ---To 1000 samples------
    x_train1000, x_test1000, y_train1000, y_test1000 = train_test_split(x_train, y_train, train_size=1000, test_size=1000)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao - 1000
    x_train1000 = pd.DataFrame(x_train1000, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    x_test1000 = pd.DataFrame(x_test1000, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    y_train1000 = pd.DataFrame(y_train1000, columns = ['TimeStamp', 'DispFrames'])
    y_test1000 = pd.DataFrame(y_test1000, columns = ['TimeStamp', 'DispFrames'])

    regr = linear_model.LinearRegression()
    regr.fit(x_train1000.iloc[:, x_train1000.columns != "TimeStamp"], y_train1000.iloc[:, y_train1000.columns != "TimeStamp"])
    y_pred1000 = regr.predict(x_test.iloc[:, x_train.columns != "TimeStamp"])

    nmae1000 = np.append(nmae1000, nmae(y_test, y_pred1000))

    # ---To 2520 samples (Complete sub-sets)------
    x_train2520, x_test1080, y_train2520, y_test1080 = train_test_split(csv_x, csv_y, test_size=0.30)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao - 2520
    x_train2520 = pd.DataFrame(x_train2520, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    x_test1080 = pd.DataFrame(x_test1080, columns = ['TimeStamp', 'all_idle', 'X_memused', 'proc.s', 'cswch.s', 'file.nr', 'sum_intr.s', 'ldavg.1','tcpsck', 'pgfree.s'])
    y_train2520 = pd.DataFrame(y_train2520, columns = ['TimeStamp', 'DispFrames'])
    y_test1080 = pd.DataFrame(y_test1080, columns = ['TimeStamp', 'DispFrames'])

    regr = linear_model.LinearRegression()
    regr.fit(x_train2520.iloc[:, x_train2520.columns != "TimeStamp"], y_train2520.iloc[:, y_train2520.columns != "TimeStamp"])
    y_pred1080 = regr.predict(x_test1080.iloc[:, x_test1080.columns != "TimeStamp"])

    nmae2520 = np.append(nmae2520, nmae(y_test1080, y_pred1080))

    # x = pd.DataFrame(x_train)
    # x_train50 = x.loc[random.sample(list(x.index), 50)]
    # x_train100 = x.loc[random.sample(list(x.index), 100)]
    # x_train500 = x.loc[random.sample(list(x.index), 500)]
    # x_train1000 = x.loc[random.sample(list(x.index), 1000)]
    #
    # y = pd.DataFrame(y_train)
    # y_train50 = y.loc[random.sample(list(x.index), 50)]
    # y_train100 = y.loc[random.sample(list(x.index), 100)]
    # y_train500 = y.loc[random.sample(list(x.index), 500)]
    # y_train1000 = y.loc[random.sample(list(x.index), 1000)]
    #
    # regr = linear_model.LinearRegression()
    # regr.fit(x_train50, y_train50)
    # y_pred50 = regr.predict(x_test)
    #
    # nmae50 = np.append(nmae50, mean_absolute_error(y_test, y_pred50))
    # #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred50))
    #
    #
    #
    # regr = linear_model.LinearRegression()
    # regr.fit(x_train100, y_train100)
    # y_pred100 = regr.predict(x_test)
    #
    # nmae100 = np.append(nmae100, mean_absolute_error(y_test, y_pred100))
    # #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred100))
    #
    # regr = linear_model.LinearRegression()
    # regr.fit(x_train500, y_train500)
    # y_pred500 = regr.predict(x_test)
    #
    # nmae500 = np.append(nmae500, mean_absolute_error(y_test, y_pred500))
    #
    # #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred500))
    #
    # regr = linear_model.LinearRegression()
    # regr.fit(x_train1000, y_train1000)
    # y_pred1000 = regr.predict(x_test)
    #
    # nmae1000 = np.append(nmae1000, mean_absolute_error(y_test, y_pred1000))
    #
    # #print("The Normalized Mean Absolute Error: %0.2f " % mean_absolute_error(y_test, y_pred1000))

#print(nmae50)
#print(nmae100)
#print(nmae200)
#print(nmae500)
#print(nmae1000)
#print(nmae2520)

data_to_plot = [nmae50, nmae100, nmae200, nmae500, nmae1000, nmae2520]

#NMAE for all Samples
plt.boxplot(data_to_plot)
plt.xlabel('N. M. A. E. for All Samples')
plt.xticks([1, 2, 3, 4, 5, 6], ['50', '100', '200', '500', '1000', '2520'])
plt.show()
