import matplotlib.pyplot as plt

class CarVisualizer:
    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Price')
        plt.show()
