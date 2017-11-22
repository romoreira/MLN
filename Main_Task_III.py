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

def get_dataframe():
    csv_x = pd.read_csv('X.csv', sep=',', header=None)
    csv_y = pd.read_csv('Y.csv', sep=',', header=None)

    # Esse trecho de codigo retira a primeira linha do DataFrame (que contem os nomes das colunas), cria uma novo DataFrame sem essa primeira linha,
    # depois adiciona as colunas na forma de indices
    new_header = csv_x.iloc[0]
    csv_x = csv_x[1:]
    csv_x.columns = new_header

    new_header = csv_y.iloc[0]
    csv_y = csv_y[1:]
    csv_y.columns = new_header

    for columns in csv_x.columns:
        csv_x[columns] = csv_x[columns].convert_objects(
            convert_numeric=True)  # Transformando os dados do DataFrameX para tipo Numerico

    for columns in csv_y.columns:
        csv_y[columns] = csv_y[columns].convert_objects(
            convert_numeric=True)  # Transformando os dados do DataFrameY para tipo Numerico

    return csv_x, csv_y

def dataset_headers(dataset):
    #Monto uma lista com os nomes das colunas
    return list(dataset.columns.values)

def binarize_y(y):
    # Adiciona a Y (Target) valores binarios para o SLA Conformance
    i = 0
    sla_conformance_y = np.array([])

    for i in range(len(y)):
        if y.iloc[i]['DispFrames'].astype(float) >= 18:
            sla_conformance_y = np.append(sla_conformance_y, 1)
        else:
            sla_conformance_y = np.append(sla_conformance_y, 0)
        i += 1

    return pd.DataFrame(sla_conformance_y)


#---------Task III ---------------------

csv_x, csv_y = get_dataframe()

x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train, columns=['TimeStamp','all_..idle','X..memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
x_test = pd.DataFrame(x_test, columns=['TimeStamp','all_..idle','X..memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
y_train = pd.DataFrame(y_train, columns=['TimeStamp','DispFrames'])
y_test = pd.DataFrame(y_test, columns=['TimeStamp','DispFrames'])

#print("XTrain")
#print(x_train)
#print("XTest")
#print(x_test)
#print("YTrain")
#rint(y_train)
#print("YTest")
#print(y_test)

#Binariza Y Train quanto a conformancia com SLA
y_train['SLA_Conformance'] = binarize_y(y_train)

#Binariza Y Teste quanto a conformancia com SLA
y_test['SLA_Conformance'] = binarize_y(y_test)

#print("YTrain_BIN")
#print(y_train)
#print("YTest_BIN")
#print(y_test.head())
#print(y_test.info())
#print(dataset_headers(y_test))

x_train_features = dataset_headers(x_train.iloc[:, x_train.columns != "TimeStamp"])
target = 'SLA_Conformance'

#print("X_Train_features:")
#print(x_train_features)



y_train = y_train.iloc[:,y_train.columns != "TimeStamp"]
y_train = y_train.iloc[:,y_train.columns != "DispFrames"]

y_test = y_test.iloc[:,y_test.columns != "TimeStamp"]
y_test = y_test.iloc[:,y_test.columns != "DispFrames"]

print(y_test)


y_train = column_or_1d(y_train, warn=False)

#Instantiate model
C = LogisticRegression()

# fit model
C.fit(x_train.iloc[:, x_train.columns != "TimeStamp"], y_train)

y_pred_class = C.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])

#y_pred_class = pd.DataFrame(y_pred_class)
print("Accuracy of the Classifier C = %.3f" % metrics.accuracy_score(y_test, y_pred_class))

print("Coeficients: "+np.array2string(C.coef_))

#print("BLA: %s " % y_test['SLA_Conformance'].values)

y_pred_class = pd.DataFrame(y_pred_class)

#y_test.to_csv("y_test.csv", sep='\t')
#y_pred_class.to_csv("y_pred_class.csv", sep='\t')

TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred_class).ravel()

print("TN: %d " % TN)
print("FP: %d " % FP)
print("FN: %d " % FN)
print("TP: %d " % TP)

print(metrics.confusion_matrix(y_test, y_pred_class))

m = len(y_test)
ERR = 1 - (TP.astype(float) + TN.astype(float))/m

print("Classification Error (ERR) - Logistic Regression-based = %.3f" % ERR)


#_________________________________________________________
#---------Classifier based in a Naive Method--------------
#_________________________________________________________

csv_x, csv_y = get_dataframe()

x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train, columns=['TimeStamp','all_..idle','X..memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
x_test = pd.DataFrame(x_test, columns=['TimeStamp','all_..idle','X..memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
y_train = pd.DataFrame(y_train, columns=['TimeStamp','DispFrames'])
y_test = pd.DataFrame(y_test, columns=['TimeStamp','DispFrames'])

y_train.to_csv("y_train_dispframes.csv", sep='\t')

y_train["SLA_Conformance"] = binarize_y(y_train)
y_test["SLA_Conformance"] = binarize_y(y_test)

y_train = y_train.iloc[:,y_train.columns != "TimeStamp"]
y_train = y_train.iloc[:,y_train.columns != "DispFrames"]

y_test = y_test.iloc[:,y_test.columns != "TimeStamp"]
y_test = y_test.iloc[:,y_test.columns != "DispFrames"]

#print(y_train)

#Ajuste de configuracao utilizada no metodo fit - treina o x apenasa com a saida binaria de y
y_train = column_or_1d(y_train, warn=False)

#Instantiate model
C_naive = LogisticRegression()

# fit model
C.fit(x_train.iloc[:, x_train.columns != "TimeStamp"], y_train)

qtd_uns = 0
qtd_zeros = 0
i = 0

for i in range(len(y_train)):
    if(y_train[i] == 1):
        qtd_uns += 1

    else:
        qtd_zeros += 1
y_train = pd.DataFrame(y_train)
y_train.to_csv("y_train.csv", sep='\t')

print("Qtd. de Uns: %d" % qtd_uns)
print("Qtd. de Zeros: %d" % qtd_zeros)

p = qtd_uns / len(y_train)
y_pred_class = np.array([])
for _ in range(len(x_test)):
    #print("Probability")
    #print(qtd_uns/len(y_train))
    #print("Choice")
    #print(np.random.binomial(1, p))
    y_pred_class = np.append(y_pred_class, np.random.binomial(1, p))

y_pred_class = pd. DataFrame(y_pred_class)
TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred_class).ravel()
print("TN (Naive): %d " % TN)
print("FP (Naive): %d " % FP)
print("FN (Naive): %d " % FN)
print("TP (Naive): %d " % TP)

print(metrics.confusion_matrix(y_test, y_pred_class))

m = len(y_test)
ERR = 1 - (TP.astype(float) + TN.astype(float))/m

print("Classification Error (ERR) - Naive-based = %.3f" % ERR)

#_________________________________________________________
#---------Build a new Classifier--------------
#_________________________________________________________

csv_x, csv_y = get_dataframe()

x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, test_size=0.30)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train = pd.DataFrame(x_train, columns=['TimeStamp','all_..idle','X..memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
x_test = pd.DataFrame(x_test, columns=['TimeStamp','all_..idle','X..memused','proc.s','cswch.s','file.nr','sum_intr.s','ldavg.1','tcpsck','pgfree.s'])
y_train = pd.DataFrame(y_train, columns=['TimeStamp','DispFrames'])
y_test = pd.DataFrame(y_test, columns=['TimeStamp','DispFrames'])

#Binarizo o y_test
y_test["SLA_Conformance"] = binarize_y(y_test)
y_test = y_test.iloc[:,y_test.columns != "TimeStamp"]
y_test = y_test.iloc[:,y_test.columns != "DispFrames"]

regr = linear_model.LinearRegression()
regr.fit(x_train.iloc[:, x_train.columns != "TimeStamp"], y_train.iloc[:, y_train.columns != "TimeStamp"])
y_pred = regr.predict(x_test.iloc[:, x_test.columns != "TimeStamp"])

#Binarizar a saida da prdicao baseada em Regressao para montar a matriz de confusao
y_pred = pd.DataFrame(y_pred)
sla_pred_bin = np.array([])
i = 0
for i in range(len(y_pred)):
    if (y_pred.iloc[i].astype(float) >= 18).bool():
        sla_pred_bin = np.append(sla_pred_bin, 1)
        i += 1
    else:
        sla_pred_bin = np.append(sla_pred_bin, 0)
        i += 1

sla_pred_bin = pd.DataFrame(sla_pred_bin)
sla_pred_bin['SLA_Conformance'] = sla_pred_bin

#print(sla_pred_bin.shape)
sla_pred_bin = column_or_1d(sla_pred_bin['SLA_Conformance'], warn=False)
#print(sla_pred_bin.shape)

#print(y_test.shape)
y_test = column_or_1d(y_test['SLA_Conformance'], warn=False)
#print(y_test.shape)

TN, FP, FN, TP = metrics.confusion_matrix(y_test, sla_pred_bin).ravel()
print("TN (New Classifier): %d " % TN)
print("FP (New Classifier): %d " % FP)
print("FN (New Classifier): %d " % FN)
print("TP (New Classifier): %d " % TP)

print(metrics.confusion_matrix(y_test, sla_pred_bin))

m = len(y_test)
ERR = 1 - (TP.astype(float) + TN.astype(float))/m

print("Classification Error (ERR) - New Classifier (Linear Regression-based) = %.3f" % ERR)