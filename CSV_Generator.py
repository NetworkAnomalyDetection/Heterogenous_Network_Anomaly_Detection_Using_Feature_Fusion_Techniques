#Code to create seperate CSVs from the main, 'all_data.csv' having equal tuples of all the types of attacks as in 'Label' comlumn of the dataset


import pandas as pd
import random

# Load the CSV file
file_path = "all_data.csv"  # Replace with the correct path to your CSV file
df = pd.read_csv(file_path)

# Group by the 'Label' column and select up to 100 rows for each distinct value
df_sampled = df.groupby('Label').apply(lambda group: group.sample(n=min(len(group), 100), random_state=42))

# Reset the index after grouping
df_sampled = df_sampled.reset_index(drop=True)

# Shuffle the rows randomly
df_shuffled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the result to a new CSV file
output_file = "sampled_and_shuffled_data.csv"  # Replace with the desired output file name
df_shuffled.to_csv(output_file, index=False)

print(f"Sampled and shuffled data saved to {output_file}")
