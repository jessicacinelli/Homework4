#
#       Feedforward neural network for binary classification -
#
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#

import numpy as np
import pandas as pd
import matplotlib as mpl
import tensorflow as tf
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras import regularizers, initializers


tf.random.set_seed(123)

class FeedforwardNN():
    #costruttore
    def __init__(self, input_dim):
            #istanziamo un input layer
            #Input proviene da Keras e serve a definire un input layer
            input_layer= Input (shape=(input_dim,))
            layer = Dense(16, activation='relu', kernel_initializer=initializers.RandomNormal() )(input_layer)
            #layer = Dense( 128, activation='relu', kernel_initializer=initializers.RandomNormal() )( layer )
          #  layer = Dense( 32, activation='relu', kernel_initializer=initializers.RandomNormal() )( input_layer )
           # layer = Dense( 32, activation='relu', kernel_initializer=initializers.RandomNormal() )( layer )
          #  layer = Dense( 16, activation='relu', kernel_initializer=initializers.RandomNormal() )( layer )
            layer = Dense( 8, activation='relu', kernel_initializer=initializers.RandomNormal() )( layer )
            layer = Dense( 2, activation='relu', kernel_initializer=initializers.RandomNormal() )( layer )

            output_layer = Activation( activation='softmax' ) (layer)

            #definiamo il modello e passiamo input e output layer
            self.classifier = Model(inputs = input_layer, outputs = output_layer)

    def summary(self, ):
        self.classifier.summary()

    def train(self, x, y):

        epochs=200
        batch_size=1024
        validation_split=0.1

        self.classifier.compile(optimizer='rmsprop', loss = 'categorical_crossentropy')
       # self.classifier.compile( optimizer='sgd', loss='categorical_crossentropy' )
        #2.
        # training vero e proprio
        #epoche: quante volte gli facciamo vedere il traninig test
        #batch_size: faccio l'aggiornamento ogni 2048
        #shuffle = True -> gli esempi vengono presi in ordine random. Serve a non polarizzare in base alla sequenza.
        #verbose riguara l'andamento del training
        # fit restituisce un po di informazioni di debug
        history= self.classifier.fit(x,y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True, verbose =2)



        # -----------------------------------------------#
        #           instructor-provided code            #
        # -----------------------------------------------#
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()

        df_history = pd.DataFrame(history.history)
        return df_history

    def predict ( self, x_evaluation ):
        #1.
        # raccolta delel predizioni

        predictions= self.classifier.predict(x_evaluation)
        print( predictions )
        #2.
        # confronto delle colonne
        outcome = predictions[:,0]>predictions[:,1]

        return outcome
