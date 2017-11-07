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
nmae50 = np.array([])
nmae100 = np.array([])
nmae500 = np.array([])
nmae1000 = np.array([])
nmae2520 = np.array([])
for _ in range(50):

    x_train, x_test, y_train, y_test = train_test_split(csv_x, csv_y, train_size=0.30, random_state=1)

    # Seto novamente a configuracao de DataFrame para nao perder a dimensao
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    x = pd.DataFrame(x_train)



    #Seto novamente a configuracao de DataFrame para nao perder a dimensao
    x_train50 = pd.DataFrame(x_train50)
    x_test50 = pd.DataFrame(x_test50)
    y_train50 = pd.DataFrame(y_train50)
    y_test50 = pd.DataFrame(y_test50)

    regr = linear_model.LinearRegression()
    regr.fit(x_train50, y_train50)
    y_pred50 = regr.predict(x_test50)

    # The Normalized mean Absolute Error
    print(mean_absolute_error(y_test50, y_pred50))
    nmae50 = np.append(nmae50, mean_absolute_error(y_test50, y_pred50))

print(nmae50)

#For 100 times
x_train100, x_test100, y_train100, y_test100 = train_test_split(x_train, y_train, train_size=100, random_state=1)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train100 = pd.DataFrame(x_train100)
x_test100 = pd.DataFrame(x_test100)
y_train100 = pd.DataFrame(y_train100)
y_test100 = pd.DataFrame(y_test100)

#For 200 times
x_train200, x_test200, y_train200, y_test200 = train_test_split(x_train, y_train, train_size=200, random_state=1)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train200 = pd.DataFrame(x_train200)
x_test200 = pd.DataFrame(x_test200)
y_train200 = pd.DataFrame(y_train200)
y_test200 = pd.DataFrame(y_test200)

#For 500 times
x_train500, x_test500, y_train500, y_test500 = train_test_split(x_train, y_train, train_size=500, random_state=1)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train500 = pd.DataFrame(x_train500)
x_test500 = pd.DataFrame(x_test500)
y_train500 = pd.DataFrame(y_train500)
y_test500 = pd.DataFrame(y_test500)

#For 1000 times
x_train1000, x_test1000, y_train1000, y_test1000 = train_test_split(x_train, y_train, train_size=1000, random_state=1)

#Seto novamente a configuracao de DataFrame para nao perder a dimensao
x_train1000 = pd.DataFrame(x_train1000)
x_test1000 = pd.DataFrame(x_test1000)
y_train1000 = pd.DataFrame(y_train1000)
y_test1000 = pd.DataFrame(y_test1000)