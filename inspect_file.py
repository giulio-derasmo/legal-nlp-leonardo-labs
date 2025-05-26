from datasets import load_dataset
import pandas as pd
import os

# Load dataset 
print('load dataset')

print(os.getcwd())

data_path = '../data'
train_filename = 'train.csv'

dataset = pd.read_csv(os.path.join(data_path, train_filename))
print(dataset)