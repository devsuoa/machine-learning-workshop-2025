# same process as mlp.py but using tensorflow, implement the code below
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

df = pd.read_csv('data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimiser, loss='mse')
model.fit(x_train, y_train, epochs=50)
y_pred = model.predict(x_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}')