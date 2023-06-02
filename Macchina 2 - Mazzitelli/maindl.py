from utils import  transformData, evaluatePerformance
from FeedforwardNN import FeedforwardNN
from AutoEncoder import AutoEncoder

train = 'hw4Data\\TRAIN.csv'
train_ae = 'hw4Data\\TRAIN_AE.csv'
validation = 'hw4Data\\VALIDATION.csv'
test = 'hw4Data\\TEST.csv'

x_train, y_train, L_train, x_train_ae, y_train_ae, L_train_ae, x_val, y_val, L_val, x_test, y_test, L_test = transformData(train, train_ae, validation, test)

# print("x_train = "+str(x_train.shape))
# print("y_train = "+str(y_train.shape))

input_dim = x_train.shape[1]

"""
ffnn = FeedforwardNN(input_dim=input_dim)
ffnn.summary()

ffnn.train(x_train, y_train)
outcome = ffnn.predict(x_test)
evaluatePerformance(outcome, L_test)

"""
ae=AutoEncoder(input_dim = input_dim)
ae.summary()

#l'autoencoder viene forzato per ricostruire gli ingressi
#passiamo come x e y sempre lo stesso df
ae.train(x_train_ae, x_train_ae)

outcome=ae.predict(x_test)
evaluatePerformance(outcome, L_test)

ae.plot_reconstruction_error(x_test, L_test)
