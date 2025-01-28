#Train all the 3 ML models on all the classes (12 attacks)
#Saves a single pkl file and all the classification reports and confusion matrices in a new folder
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Path definitions to all required files and folders
csv_files = "./all_data.csv"
path = ""
repetition = 10

# Path to result storing directory
results_dir = "Results_of_Training"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Columns to use in the dataset
usecols = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", 
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", 
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", 
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", 
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", 
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", 
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", 
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", 
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", 
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", 
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", 
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", 
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", 
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", 'Label'
]

attack_names = [
    "BENIGN","Bot","DDoS","DoS GoldenEye","DoS Hulk","DoS Slowhttptest","DoS slowloris","FTP-Patator","Heartbleed","Infiltration","PortScan","SSH-Patator","Web Attack � Brute Force","Web Attack � Sql Injection","Web Attack � XSS"
]

ml_list = {
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=50, max_features='sqrt'),
    "ID3": DecisionTreeClassifier(criterion='entropy'),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

training_parameters = [
    "Bwd Packet Length Min","Flow IAT Std","Fwd Packet Length Std","Flow IAT Min","Bwd Packet Length Std","Bwd Packet Length Mean"
]

seconds = time.time()

print('%-17s %-17s  %-15s ' % ("File", "ML algorithm", "Accuracy"))
feature_list = usecols
df = pd.read_csv(path + csv_files, usecols=feature_list)
df = df.fillna(0)

# Convert labels to numeric and map unique labels to attack names dynamically
df["Label"] = df["Label"].astype('category').cat.codes
unique_labels = sorted(df["Label"].unique())
label_mapping = {label: attack_names[label] for label in unique_labels}

y = df["Label"]
X = df[training_parameters]

for ii in ml_list:
    accuracy = []
    classification_reports = []
    confusion_matrices = []
    t_time = []
    for i in range(repetition):
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=i)
        clf = ml_list[ii]
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Dynamically filter target names for only the labels in the test set
        test_unique_labels = sorted(list(set(y_test) | set(y_pred)))
        test_label_mapping = {label: label_mapping[label] for label in test_unique_labels}

        # Generate classification report
        report = classification_report(
            y_test, y_pred, target_names=[test_label_mapping[label] for label in test_unique_labels], zero_division=0
        )

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=test_unique_labels)

        # Store results
        accuracy.append(acc)
        classification_reports.append(report)
        confusion_matrices.append(cm)
        t_time.append(time.time() - start_time)

    # Save the final model for the algorithm
    joblib.dump(clf, os.path.join(results_dir, f"{ii}_model.pkl"))

    # Average accuracy
    avg_accuracy = np.mean(accuracy)
    print('%-17s %-17s  %-15s' % (csv_files[0:-4], ii, str(round(avg_accuracy, 2))))
    print(f"Classification Report for {ii} on {csv_files[0:-4]}:")
    print(classification_reports[0])

    # Save classification report
    with open(os.path.join(results_dir, f"{ii}_classification_report.txt"), "w") as f:
        f.write(classification_reports[0])

    # Plot confusion matrix
    cm = confusion_matrices[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='g', cmap='Blues',
        xticklabels=[test_label_mapping[label] for label in test_unique_labels],
        yticklabels=[test_label_mapping[label] for label in test_unique_labels]
    )
    plt.title(f"Confusion Matrix for {ii} on {csv_files[0:-4]}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(results_dir, f"{ii}_confusion_matrix.png"))
    plt.close()

print("Mission accomplished!")
print("Total operation time: = ", time.time() - seconds, "seconds")