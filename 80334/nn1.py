#Importing Numpy
import numpy as np


#Importing Pandas
import pandas as pd


#Importing Matpplotlib
import matplotlib.pyplot as plt



#Importing Tensorflow
import tensorflow as tf


#Importing Keras backend
import tensorflow.keras.backend as K



#Print Tensorflow Version
print(tf.__version__)



#Importing Warnings
import warnings

#Ignoring Warnings
warnings.filterwarnings("ignore")


#Function for getting F1-Score
def get_precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return precision


#Read the dataset into a dataframe
df = pd.read_csv('amloydset.csv')


#Mapping of Food Item 1
df["LeafyGreens"] = df["LeafyGreens"].map({'LessThan6Servings':1 ,'6Servings':2,'MoreThan6Servings':3})


#Mapping of Food Item 2
df["Champagne"] = df["Champagne"].map({'NoChampagne':1 ,'Around3Glasses':2,'MoreThan6Glasses':3})


#Mapping of Food Item 3
df["Vegetables"] = df["Vegetables"].map({'Rarely':1 ,'Regularly':2,'Daily':3})


#Mapping of Food Item 4
df["Walnuts"] = df["Walnuts"].map({'LessThanHalfCup':1 ,'HalfCup':2,'MoreThanHalfCup':3})


#Mapping of Food Item 5
df["Onions"] = df["Onions"].map({'NotSpecific':1 ,'Regularly':2,'Compulsorily':3})


#Mapping of Food Item 6
df["Coffee"] = df["Coffee"].map({'NoToLowCoffee':1 ,'Around2cups':2,'MoreThan2cups':3})


#Mapping of Food Item 7
df["Turmeric"] = df["Turmeric"].map({'NotSpecific':1 ,'RegularUsage':2,'DailyUsage':3})


#Mapping of Food Item 8
df["Cinnamon"] = df["Cinnamon"].map({'NoCinnamon':1 ,'LowCinnamon':2,'CinnamonDaily':3})


#Mapping of Food Item 9
df["FattyFish"] = df["FattyFish"].map({'NoToLow':1 ,'3to5Servings':2,'MoreThan5':3})


#Mapping of Food Item 10
df["Berries"] = df["Berries"].map({'LessThan2cups':1 ,'3or4cups':2,'Above4cups':3})

#Mapping of SleepQuality
df["AmloydProteins"] = df["AmloydProteins"].map({'LowAmloyds':0 ,'NeededAmloyds':1,'MoreAmloyds':2})


#Creation of data as numpy array
data = df[["LeafyGreens","Champagne","Vegetables","Walnuts","Onions","Coffee","Turmeric","Cinnamon","FattyFish","Berries","AmloydProteins"]].to_numpy()


#All columns except last column are considered as inputs
inputs = data[:,:-1]


#Last Column is considered as outputs
outputs = data[:, -1]



#First Thousand rows are considered for training.
training_data = inputs[:600]


#Training labels are set to the last column values of first thousand rows
training_labels = outputs[:600]



#Remaining Rows, Beyond 600 are considered for testing
test_data = inputs[600:]


#Testing labels are set to the last column values of remaining rows
test_labels = outputs[600:]


#Tensorflow Initiation
tf.keras.backend.clear_session()




#Configure the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(32, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
									
									
#Comiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



#Creation of the model
model.fit(training_data, training_labels, epochs=150)


#Print Models Loss and Accuracy
print("Models Loss and Accuracy are",model.evaluate(test_data, test_labels))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")