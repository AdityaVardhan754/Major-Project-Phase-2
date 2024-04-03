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
accuracy = accuracy_score(testing_outputs, predictions)



def get_user_input():
    food_data = {}
    # Prompt user for each food item
    food_data["LeafyGreens"] = input("Enter Leafy Greens intake (1 - Less than 6 servings, 2 - 6 servings, 3 - More than 6 servings): ")
    food_data["Champagne"] = input("Enter Champagne intake (1 - No Champagne, 2 - Around 3 Glasses, 3 - More than 6 Glasses): ")
    food_data["Vegetables"] = input("Enter Vegetables intake (1 - Rarely, 2 - Regularly, 3 - Daily): ")
    food_data["Walnuts"] = input("Enter Walnuts intake (1 - LessThanHalfCup, 2 - HalfCup, 3 - MoreThanHalfCup): ")
    food_data["Onions"] = input("Enter Onions intake (1 - NotSpecific, 2 - Regularly, 3 - Compulsorily): ")
    food_data["Coffee"] = input("Enter Coffee intake (1 - NoToLowCoffee, 2 - Around2cups, 3 - MoreThan2cups): ")
    food_data["Turmeric"] = input("Enter Turmeric intake (1 - NotSpecific, 2 - RegularUsage, 3 - DailyUsage): ")
    food_data["Cinnamon"] = input("Enter Cinnamon intake (1 - NoCinnamon, 2 - LowCinnamon, 3 - CinnamonDaily): ")
    food_data["FattyFish"] = input("Enter FattyFish intake (1 - NoToLow, 2 - 3to5Servings, 3 - MoreThan5): ")
    food_data["Berries"] = input("Enter Berries intake (1 - LessThan2cups, 2 - 3or4cups, 3 - Above4cups): ")
    return food_data

def get_user_input_as_datapoint(user_data):
    mapping_dictionary = {
        "LeafyGreens": {"1": 1, "2": 2, "3": 3},  # Direct numerical mapping
        "Champagne": {"1": 1, "2": 2, "3": 3},
        "Vegetables": {'1': 1, '2': 2, '3': 3},
        "Walnuts": {'1': 1, '2': 2, '3': 3},
        "Onions": {'1': 1, '2': 2, '3': 3},
        "Coffee": {'1': 1, '2': 2, '3': 3},
        "Turmeric": {'1': 1, '2': 2, '3': 3},
        "Cinnamon": {'1': 1, '2': 2, '3': 3},
        "FattyFish": {'1': 1, '2': 2, '3': 3},
        "Berries": {'1': 1, '2': 2, '3': 3},
    }
    numerical_data = []
    for key, value in user_data.items():
        numerical_data.append(mapping_dictionary[key][value])
    return np.array([numerical_data])


def predict_gbc(user_data):
    datapoint = get_user_input_as_datapoint(user_data)
    prediction = classifier.predict(datapoint)[0]
    amloyds_mapping = {0: "LowAmloyds", 1: "NeededAmloyds", 2: "MoreAmloyds"}
    prediction_label = amloyds_mapping[prediction]
    print("GBC Model Prediction:", prediction_label)



if __name__ == "__main__":
    # Train and evaluate the GBC model
    classifier.fit(training_inputs, training_outputs)
    predictions = classifier.predict(testing_inputs)
    accuracy = accuracy_score(testing_outputs, predictions)
    print("Gradient Boosting Classifier Accuracy:", accuracy)

    # Get user input and make prediction
    user_data = get_user_input()
    predict_gbc(user_data)
