#To understand the statistics of the consolidated dataset
##  all_data.csv files is required for the operation of the program.
##  all_data.csv file must be located in the same directory as the program.



##  The purpose of this program is to provide statistics about the data contained in the dataset.
##  Considering that some of the data are very large and some of them are very small, the graphics are created in three separate groups, so that all data can be seen:
##          big: labels with more than 11000 numbers
##          medium: labels with numbers between 600 and 11000
##          small: labels with fewer than 600 numbers
##  
##  In the last graphics, the rates of all attacks and normal behaviors are given.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Start timer
seconds = time.time()

# Graph creation function
def graph(objects, performance, x_label, y_label):
    y_pos = np.arange(len(objects))
    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel(x_label)
    plt.title(y_label)
    plt.show()

# Load dataset
df = pd.read_csv('all_data.csv', usecols=["Label"])
print(df.iloc[:, 0].value_counts())
a = df.iloc[:, 0].value_counts()

# Separate labels and values into different categories
key = a.keys()
values = a.values
small_labels = []
small_values = []
big_labels = []
big_values = []
medium_labels = []
medium_values = []
attack = 0
benign = 0

# Group attacks into categories based on values
for i in range(len(values)):
    if values[i] > 11000:
        big_labels.append(str(key[i]))
        big_values.append(values[i])
    elif values[i] < 600:
        small_labels.append(str(key[i]))
        small_values.append(values[i])
    else:
        medium_labels.append(str(key[i]))
        medium_values.append(values[i])

    if str(key[i]) == "BENIGN":
        benign += values[i]
    else:
        attack += values[i]

key = [benign, attack]

# Create charts
labels = [
    "BENIGN %" + str(round(benign / (benign + attack), 2) * 100),
    "ATTACK %" + str(round(attack / (benign + attack), 2) * 100)
]
graph(big_labels, big_values, "Numbers", "Attacks Labels - High-number group")
graph(medium_labels, medium_values, "Numbers", "Attacks Labels - Medium-number group")
graph(small_labels, small_values, "Numbers", "Attacks Labels - Small-number group")
graph(labels, key, "Numbers", "Attack and Benign Percentage")

# Print completion message and total time
print("Mission accomplished!")
print("Total operation time: =", time.time() - seconds, "seconds")
