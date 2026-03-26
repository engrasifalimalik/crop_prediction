import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Yield")
    plt.savefig("results/prediction_plot.png")
    plt.show()
