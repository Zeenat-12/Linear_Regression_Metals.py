Resistance vs Temperature: Linear Regression with Interaction Terms

A small machine learning project that predicts the electrical resistance of metal wires from their material and temperature. It compares two linear regression models to show why interaction terms matter.

Overview

The electrical resistance of a metal rises with temperature, following:

R = R20 * (1 + alpha * (T - 20))
R20 is the resistance at 20 °C
alpha is the temperature coefficient of the material
T is the temperature in °C

The project generates synthetic measurements (with random noise) for four metals and trains two models on them.

Material	R20 (ohm)	alpha
Copper	0.168	0.0039
Aluminum	0.282	0.0043
Iron	0.971	0.0065
Tungsten	0.560	0.0045
The Two Models

Model 1: One shared slope Uses temperature plus one-hot encoded material columns. Each material gets its own starting value (intercept), but all materials share a single slope, so the fitted lines are parallel. This is too simple, because the true slope (R20 * alpha) is different for every metal.

Model 2: Slope per material Adds interaction features (material * temperature). Each material now gets its own intercept and its own slope, which matches the underlying physics and gives a much better fit.

What the Code Does
Generates 200 data points (50 per material) with random temperatures between 0 and 200 °C and Gaussian measurement noise.
One-hot encodes the material column.
Splits the data into 80% training and 20% testing sets.
Trains Model 1 (shared slope) with LinearRegression.
Builds interaction features and trains Model 2 (slope per material).
Compares both models using R² and MAE on the test set.
Plots both models over the data points.
Requirements
Python 3.8+
numpy
pandas
matplotlib
scikit-learn

Install them with:

bash
pip install numpy pandas matplotlib scikit-learn
Usage

Run the script:

bash
python resistance_temperature_regression.py

Or open the code in Google Colab or Jupyter Notebook and run the cells from top to bottom.

Output
Metrics: R² and MAE for both models, printed to the console.
Example predictions: resistance of copper and iron at 100 °C.
Graph 1: "One shared slope". The lines are parallel and miss the tilt of the data, especially for iron and copper.
Graph 2: "Slope per material". Each line follows its own group of points.

Model 2 should show a higher R² (closer to 1) and a lower MAE than Model 1.

Key Takeaway

When the effect of one variable (temperature) depends on another variable (material), a plain linear model with one-hot columns is not enough. Adding interaction terms lets each category have its own slope and fixes the fit.

Notes
The data is synthetic and generated with np.random.seed(42), so results are reproducible.
Logistic regression is not used here because it is for classification. Resistance is a continuous value, so linear regression is the right choice.
