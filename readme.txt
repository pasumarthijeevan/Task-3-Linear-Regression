# House Price Prediction using Linear Regression

## Project Overview
This project demonstrates the implementation of **Simple and Multiple Linear Regression** using Python and Scikit-learn. The model predicts house prices based on features such as area, number of bedrooms, and number of bathrooms.

## Objective
- Understand Simple and Multiple Linear Regression.
- Train a Linear Regression model.
- Evaluate the model using MAE, MSE, and R² Score.
- Visualize the regression results.
- Interpret model coefficients.

## Technologies Used
- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```
Task-3-Linear-Regression/
│── dataset/
│   └── house_price.csv
│── output/
│── screenshots/
│── linear_regression.py
│── README.md
└── requirements.txt
```

## Dataset
The dataset contains information about houses, including:
- Area
- Bedrooms
- Bathrooms
- Price

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## How to Run

1. Open the project folder in VS Code.
2. Open the terminal.
3. Install the required packages:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
4. Run the project:
   ```bash
   python linear_regression.py
   ```

## Evaluation Metrics

The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Output

The program displays:
- Dataset information
- Missing values
- Model coefficients
- MAE
- MSE
- R² Score
- Actual vs Predicted graph
- Simple Linear Regression graph

## Project Features

- Data preprocessing
- Train-test split
- Simple Linear Regression
- Multiple Linear Regression
- Model evaluation
- Data visualization
- Coefficient interpretation

## Future Improvements

- Feature scaling
- Hyperparameter tuning
- Cross-validation
- Deployment using Flask or Streamlit

## Conclusion

This project successfully implements both Simple and Multiple Linear Regression for house price prediction. The trained model demonstrates how different house features influence price and is evaluated using standard regression metrics. The project provides a solid foundation for understanding regression analysis and predictive modeling in machine learning.
