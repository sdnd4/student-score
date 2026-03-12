import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset simple
data = {
    "hours_studied": [1,2,3,4,5,6,7,8],
    "score": [30,35,50,55,65,70,80,90]
}

df = pd.DataFrame(data)

X = df[['hours_studied']]
y = df['score']

# split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# modèle ML
model = LinearRegression()
model.fit(X_train,y_train)

# prédictions
predictions = model.predict(X_test)

# évaluation
mse = mean_squared_error(y_test,predictions)

print("Model trained successfully")
print("Mean Squared Error:", mse)
