from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.utils.validation import column_or_1d
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
import math
import cmath

def fit_interleaves_features(x_train, y_train, x_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    return y_pred

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
    media = 0
    nmae_resultado = 0
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
    
def get_dataframe():
    csv_x = pd.read_csv('./data/X.csv', sep=',', header=None)
    csv_y = pd.read_csv('./data/Y.csv', sep=',', header=None)

    # Esse trecho de codigo retira a primeira linha do DataFrame (que contem os nomes das colunas), cria uma novo DataFrame sem essa primeira linha,
    # depois adiciona as colunas na forma de indices
    new_header = csv_x.iloc[0]
    csv_x = csv_x[1:]
    csv_x.columns = new_header

    new_header = csv_y.iloc[0]
    csv_y = csv_y[1:]
    csv_y.columns = new_header

    csv_x['TimeStamp'] = pd.to_numeric(csv_x['TimeStamp'])
    csv_x['all_..idle'] = pd.to_numeric(csv_x['all_..idle'])
    csv_x['X..memused'] = pd.to_numeric(csv_x['X..memused'])
    csv_x['proc.s'] = pd.to_numeric(csv_x['proc.s'])
    csv_x['cswch.s'] = pd.to_numeric(csv_x['cswch.s'])
    csv_x['file.nr'] = pd.to_numeric(csv_x['file.nr'])
    csv_x['sum_intr.s'] = pd.to_numeric(csv_x['sum_intr.s'])
    csv_x['tcpsck'] = pd.to_numeric(csv_x['tcpsck'])
    csv_x['pgfree.s'] = pd.to_numeric(csv_x['pgfree.s'])

    csv_y['TimeStamp'] = pd.to_numeric(csv_y['TimeStamp'])
    csv_y['DispFrames'] = pd.to_numeric(csv_y['DispFrames'])

    return csv_x, csv_y


def dataset_headers(dataset):
    # Monto uma lista com os nomes das colunas
    return list(dataset.columns.values)


def binarize_y(y):
    # Adiciona a Y (Target) valores binarios para o SLA Conformance
    i = 0
    sla_conformance_y = np.array([])

    for i in range(len(y)):
        if y.iloc[i]['DispFrames'] >= 18:
            sla_conformance_y = np.append(sla_conformance_y, 1.0)
        else:
            sla_conformance_y = np.append(sla_conformance_y, 0.0)
        i += 1

    return sla_conformance_y


# ---------Task III ----------------------------
#_______________________________________________


csv_x, csv_y = get_dataframe()
x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30)

#---------Question 2----------------------------
#_______________________________________________


#y_train = column_or_1d(y_train, warn=False)
y_train = column_or_1d(y_train.iloc[:, y_train.columns != "TimeStamp"], warn=False)
model = linear_model.LinearRegression()
rfe = RFE(model, 3)
fit = rfe.fit(x_train.iloc[:, x_train.columns != "TimeStamp"],y_train)
n_features = fit.n_features_
selected_feature = fit.support_
feature_ranking = fit.ranking_
print("Num Features: %.f" % n_features.astype(float))
print("Selected Features: %s" %selected_feature)
print("Feature Ranking: %s"  %feature_ranking)

#Teste para conjunto com 1 caracteristica
x_train_bkp = x_train
y_train_bkp = y_train
x_test_bkp = x_test
y_test_bkp = y_test
for column in x_train:
    if(column == "TimeStamp"):
        print()
    else:
        x_train = x_train.iloc[:,x_train.columns != "TimeStamp"][column]
        x_train = pd.DataFrame(x_train, columns = [column])
        
        x_test = x_test.iloc[:,x_test.columns != "TimeStamp"][column]
        x_test = pd.DataFrame(x_test, columns = [column])
        
        
        print("NMAE for ['%s'] is: %.3f" % (column, nmae(y_test, fit_interleaves_features(x_train, y_train, x_test))))
        #print(fit_interleaves_features(x_train, y_train.iloc[:,y_train.columns != "TimeStamp"], x_test))   
        x_train = x_train_bkp
        x_test = x_test_bkp

##Teste para conjunto com 2 caracteristicas
x_train = x_train_bkp
x_test = x_test_bkp
features = list(x_train.columns)

#Crio um numpy array para armazenar as NMAE para posteriormente plotar o Histogram
nmae_array = np.array([])

i = 1
while i < len(features):
    j = i+1
    while j < len(features):

        x_train = x_train.iloc[:, [i,j]]
        
        x_test = x_test.iloc[:, [i,j]]
        
        nmae_array = np.append(nmae_array, nmae(y_test, fit_interleaves_features(x_train, y_train, x_test)))
        
        ##print(nmae_array)
        
        print("NMAE for ['%s','%s] is: %.4f" % (features[i], features[j], nmae(y_test, fit_interleaves_features(x_train, y_train, x_test))))
        j += 1
        x_train = x_train_bkp
        x_test = x_test_bkp
    i += 1
    
#NAMAE Histogram
plt.hist(nmae_array, 4, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('% NMAE')
plt.title("NMAE for all Sets")
plt.show()


# ---------Task III ---------------------

csv_x, csv_y = get_dataframe()

x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30)

#_______________________________________________
#---------Question 3----------------------------
#_______________________________________________

features= np.array([])
for column in x_train:
    features= np.append(features,column)
 
correlation_array = np.float32([])
i = 1
while i < len(features):
    x_column = np.array(x_train[features[i]])
    y_train = y_train.iloc[:, y_train.columns == 'DispFrames']
    y = np.array(y_train)
    y_values = column_or_1d(y, warn=False)
  
    #print(pearsonr(x_column.astype(float), y_values.astype(float)))
    #print(pearsonr(x_column.astype(float), y_values.astype(float))[0])
    correlation_array = np.append(correlation_array, pearsonr(x_column.astype(float), y_values.astype(float))[0].astype(float))
    i += 1

temp = np.column_stack((features[1:],correlation_array))
correlation_set = pd.DataFrame(temp,columns=['Feature','Correlation'])
#print(correlation_set)

i = 0
correlation_square = np.float32([])
while i < len(correlation_set):
    correlation_square = np.append(correlation_square, float(correlation_set.loc[i, 'Correlation'])*float(correlation_set.loc[i, 'Correlation']))
    i += 1

correlation_set['R2'] = pd.to_numeric(correlation_square)
#print(correlation_set)
correlation_set = correlation_set.sort_values(by='R2', ascending=False)
print(correlation_set)

features_list = list(correlation_set['Feature'])
#print(features_list)

x_train = x_train[features_list]
x_test = x_test[features_list]

x_train_bkp = x_train
y_train_bkp = y_train
x_test_bkp = x_test
y_test_bkp = y_test

#Crio um numpy array para armazenar as NMAE para posteriormente plotar o Histogram
nmae_array = np.array([])

#print(x_train.iloc[:,0:1])

#print(features_list)

i = 1
while i <= len(features_list):

    x_train = x_train.iloc[:, 0:i]
    x_test = x_test.iloc[:, 0:i]
    
    nmae_array = np.append(nmae_array, nmae(y_test, fit_interleaves_features(x_train, y_train, x_test)))
        
    ##print(nmae_array)
        
    print("NMAE for %s is: %.7f" % (features_list[:i], nmae(y_test, fit_interleaves_features(x_train, y_train, x_test))))
    j += 1
    x_train = x_train_bkp
    x_test = x_test_bkp
    i += 1
    
plt.plot(['K1','K1..2','K1..3','K1..4','K1..5','K1..6','K1..7','K1..8','K1..9'], nmae_array, '-g*')
plt.show()
