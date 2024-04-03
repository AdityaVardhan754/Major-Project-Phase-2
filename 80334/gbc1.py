import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('amloydset.csv')
df["LeafyGreens"] = df["LeafyGreens"].map({'LessThan6Servings':1 ,'6Servings':2,'MoreThan6Servings':3})
df["Champagne"] = df["Champagne"].map({'NoChampagne':1 ,'Around3Glasses':2,'MoreThan6Glasses':3})
df["Vegetables"] = df["Vegetables"].map({'Rarely':1 ,'Regularly':2,'Daily':3})
df["Walnuts"] = df["Walnuts"].map({'LessThanHalfCup':1 ,'HalfCup':2,'MoreThanHalfCup':3})
df["Onions"] = df["Onions"].map({'NotSpecific':1 ,'Regularly':2,'Compulsorily':3})
df["Coffee"] = df["Coffee"].map({'NoToLowCoffee':1 ,'Around2cups':2,'MoreThan2cups':3})
df["Turmeric"] = df["Turmeric"].map({'NotSpecific':1 ,'RegularUsage':2,'DailyUsage':3})
df["Cinnamon"] = df["Cinnamon"].map({'NoCinnamon':1 ,'LowCinnamon':2,'CinnamonDaily':3})
df["FattyFish"] = df["FattyFish"].map({'NoToLow':1 ,'3to5Servings':2,'MoreThan5':3})
df["Berries"] = df["Berries"].map({'LessThan2cups':1 ,'3or4cups':2,'Above4cups':3})
df["AmloydProteins"] = df["AmloydProteins"].map({'LowAmloyds':0 ,'NeededAmloyds':1,'MoreAmloyds':2})
data = df[["LeafyGreens","Champagne","Vegetables","Walnuts","Onions","Coffee","Turmeric","Cinnamon","FattyFish","Berries","AmloydProteins"]].to_numpy()
inputs = data[:,:-1]
outputs = data[:, -1]
training_inputs = inputs[:600]
training_outputs = outputs[:600]
testing_inputs = inputs[600:]
testing_outputs = outputs[600:]
classifier = GradientBoostingClassifier()
classifier.fit(training_inputs, training_outputs)
predictions = classifier.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of Gradient Boosting Classifier Classifier on testing data is: " + str(accuracy))