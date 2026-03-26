import joblib
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split

df = load_data("../data/crop_data.csv")
df = preprocess_data(df)

X = df.drop("yield", axis=1)
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = joblib.load("../models/model.pkl")

predictions = model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, predictions, squared=False))
print("R2 Score:", r2_score(y_test, predictions))
