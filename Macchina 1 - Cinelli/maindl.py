#
#       Main program to train / validate / test the models
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#


from utils import  transformData, evaluatePerformance
from FeedforwardNN import FeedforwardNN
from AutoEncoder import AutoEncoder

train_ae = '.\\hw4Data\\TRAIN_AE.csv'
train = '.\\hw4Data\\TRAIN.csv'
validation = '.\\hw4Data\\VALIDATION.csv'
test = '.\\hw4Data\\TEST.csv'

x_train, y_train, l_train, x_val, y_val, l_val, x_test, y_test, l_test = transformData(train, validation, test)

print("x_train = " +str(x_train.shape))
print("y_train = " +str(y_train.shape))

print("x_val = " +str(x_val.shape))
print("y_val = " +str(y_val.shape))

#prendiamo input_dim che rappresenta il numero di colonne
input_dim= x_train.shape[1]

'''
#istanziamo il modello
ffnn = FeedforwardNN(input_dim=input_dim)

ffnn.summary()

#addestramento del modello
ffnn.train(x_train, y_train)

#print delle predictions
outcome=ffnn.predict(x_val)

evaluatePerformance(outcome, l_val)


outcome=ffnn.predict(x_test)

evaluatePerformance(outcome, l_test)
'''
x_train, y_train, l_train, x_val, y_val, l_val, x_test, y_test, l_test = transformData(train_ae, validation, test)
input_dim= x_train.shape[1]
ae=AutoEncoder(input_dim = input_dim)
ae.summary()

#l'autoencoder viene forzato per ricostruire gli ingressi
#passiamo come x e y sempre lo stesso df
ae.train(x_train, x_train)

outcome=ae.predict(x_val)
evaluatePerformance(outcome, l_val)

ae.plot_reconstruction_error(x_val, l_val)
outcome=ae.predict(x_test)
evaluatePerformance(outcome, l_test)

ae.plot_reconstruction_error(x_test, l_test)