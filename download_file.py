from datasets import load_dataset
import pandas as pd

# Load dataset 
print('load dataset')
dataset_name = 'giulioderasmo/ita-legal-corpus'
dataset = load_dataset(dataset_name)

# Convert to pandas DataFrame
print('convert to pandas')
train_ds = dataset["train"].to_pandas()
test_ds = dataset["test"].to_pandas()

# Save to CSV in a local folder
print('save files')
train_ds.to_csv("./data/train.csv", index=False)
test_ds.to_csv("./data/test.csv", index=False)