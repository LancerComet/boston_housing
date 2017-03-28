import numpy as np
import pandas as pd

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print np.mean(data[data['RM'] > 8][data['RM'] < 9])
