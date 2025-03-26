import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'Laptop_price.csv'
df = pd.read_csv(file_path)
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)