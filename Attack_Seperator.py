#Code to uniquely identify every value for the 'Label' attribute and create a seperate csv for each of them

import random
import os
import pandas as pd
import time

# Start timer
seconds = time.time()

def folder(f_name):  # This function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print("The folder could not be created!")

print("This process may take 3 to 8 minutes, depending on the performance of your computer.\n\n\n")

# Headers of column
main_labels = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", 
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", 
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", 
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", 
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", 
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", 
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", 
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", 
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", 
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", 
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", 
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", 
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", 
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", 
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", 
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", 
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label", "External IP"
]
main_labels = ",".join(i for i in main_labels)

attacks = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", 
    "DoS slowloris", "FTP-Patator", "Heartbleed", "Infiltration", "PortScan", 
    "SSH-Patator", "Web Attack – Brute Force", "Web Attack – Sql Injection", 
    "Web Attack – XSS"
]
folder("./attacks/")

benign = 2359289

dict_attack = {
    "Bot": 1966,
    "DDoS": 41835,
    "DoS GoldenEye": 10293,
    "DoS Hulk": 231073,
    "DoS Slowhttptest": 5499,
    "DoS slowloris": 5796,
    "FTP-Patator": 7938,
    "Heartbleed": 11,
    "Infiltration": 36,
    "PortScan": 158930,
    "SSH-Patator": 5897,
    "Web Attack - Brute Force": 1507,
    "Web Attack - XSS": 652,
    "Web Attack - Sql Injection": 21
}

for i in dict_attack:  # In this section, a file is opened for each attack type and is recorded at a random benign flow.
    a, b = 0, 0
    with open(f"./attacks/{i}.csv", "w") as ths:
        ths.write(str(main_labels) + "\n")
        benign_num = int(benign / (dict_attack[i] * (7 / 3)))
        with open("all_data.csv", "r") as file:
            while True:
                try:
                    line = file.readline()
                    if not line:
                        break
                    line = line[:-1]
                    k = line.split(",")
                    if k[83] == "BENIGN":
                        rnd = random.randint(1, benign_num)
                        if rnd == 1:
                            ths.write(str(line) + "\n")
                            b += 1
                    if k[83] == i:
                        ths.write(str(line) + "\n")
                        a += 1
                except Exception as e:
                    print(f"Error processing line: {e}")
                    break
    print(f"{i} file is completed\n attack: {a}\n benign: {b}\n\n\n ")

# All web attack files are merged into a single file.
webs = ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"]
flag = True
for i in webs:
    df = pd.read_csv(f"./attacks/{i}.csv")
    if flag:
        df.to_csv('./attacks/Web Attack.csv', index=False)
        flag = False
    else:
        df.to_csv('./attacks/Web Attack.csv', index=False, header=False, mode="a")
    os.remove(f"./attacks/{i}.csv")

print("Mission accomplished!")
print("Operation time: ", time.time() - seconds, "seconds")
