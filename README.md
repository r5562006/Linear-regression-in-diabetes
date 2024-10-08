# Linear Regression on Diabetes Dataset

This project demonstrates the application of a multivariate linear regression model on the **Diabetes dataset** from `scikit-learn`. The goal is to predict the diabetes progression based on certain features.

## Requirements

To run the project, you need to have the following libraries installed:

- `scikit-learn`
- `numpy`

You can install these packages via `pip`:

```bash
pip install scikit-learn
pip install numpy
```

## Datasets

The project uses the Diabetes dataset provided by . This dataset contains 442 samples and 10 features. The target variable represents a quantitative measure of disease progression one year after baseline.scikit-learn

## Project structure

The project consists of the following steps:

1. Load Dataset:
Load the diabetes dataset from .scikit-learn

2. Train-Test Split:
Split the data into training and testing sets with an 80-20 ratio.

3. Model Creation:
A linear regression model is created using the class from .LinearRegressionscikit-learn

4. Model Training:
The model is trained using the training dataset.

5. Model Evaluation:
The model is evaluated by calculating the mean squared error (MSE) for both the training and testing sets.

```bash
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create a linear regression model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)

# Make predictions on training and testing sets
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# Print the mean squared error
print("Mean Squared Error (Train): %.2f" % mean_squared_error(y_train, y_pred_train))
print("Mean Squared Error (Test): %.2f" % mean_squared_error(y_test, y_pred_test))
```

## Result

The model prints out the Mean Squared Error (MSE) for both the training and testing datasets, which gives an indication of how well the model fits the data.

Example output:
```bash
Mean Squared Error (Train): 2890.57
Mean Squared Error (Test): 3440.68
```

## Code Test

If you want to write some test scripts to check whether your model is functioning correctly, here is a simple example using the unittest framework.

```bash
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
```
