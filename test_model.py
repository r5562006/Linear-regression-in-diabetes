import unittest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class TestDiabetesModel(unittest.TestCase):

    def setUp(self):
        # Load the dataset and split it
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        Y = diabetes.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2)

        # Initialize the model
        self.model = LinearRegression()

    def test_training(self):
        # Train the model and check that it fits the data
        self.model.fit(self.X_train, self.y_train)
        y_pred_train = self.model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, y_pred_train)
        self.assertTrue(mse > 0)

    def test_prediction(self):
        # Train the model and check predictions on test set
        self.model.fit(self.X_train, self.y_train)
        y_pred_test = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred_test)
        self.assertTrue(mse > 0)

if __name__ == '__main__':
    unittest.main()
