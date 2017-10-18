import pandas as pd

csv_x=pd.read_csv('X.csv', sep=',',header=None)
csv_y=pd.read_csv('Y.csv', sep=',',header=None)


#Esse trecho de codigo retira a primeira linha do DataFrame (que contem os nomes das colunas), cria uma novo DataFrame sem essa primeira linha, depois adiciona as colunas na forma de indices
new_header = csv_x.iloc[0]
csv_x = csv_x[1:]
csv_x.columns = new_header

for columns in csv_x.columns:
    #print("Coluna '"+columns+"':\n%s " % csv_x[columns].astype(float).describe())
    print("Coluna '" + columns + "':\n")
    print("Mean: %0.f" % csv_x[columns].astype(float).mean())
    print("Maximum: %0.f" % csv_x[columns].astype(float).max())
    print("Minimum: %0.f" % csv_x[columns].astype(float).min())
    print("25 Percentil %0.2f" % csv_x[columns].astype(float).quantile(.25))
    print("90 Percentil %0.2f" % csv_x[columns].astype(float).quantile(.90))
    print("Standard Deviation: %0.2f" % csv_x[columns].astype(float).std())
    print("\n")