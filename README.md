# **Resistance vs Temperature: Linear Regression with Interaction Terms**

A small machine learning project that predicts the electrical resistance of metal wires from their material and temperature. It compares two linear regression models to show why **interaction terms** matter.

## **Overview**

The electrical resistance of a metal rises with temperature, following:

```
R = R20 * (1 + alpha * (T - 20))
```

- `R20` is the resistance at 20 °C
- `alpha` is the temperature coefficient of the material
- `T` is the temperature in °C

The project generates synthetic measurements (with random noise) for four metals and trains two models on them.

| Material | R20 (ohm) | alpha |
|----------|-----------|-------|
| Copper   | 0.168     | 0.0039 |
| Aluminum | 0.282     | 0.0043 |
| Iron     | 0.971     | 0.0065 |
| Tungsten | 0.560     | 0.0045 |

## **The Two Models**

**Model 1: One shared slope**
Uses temperature plus one-hot encoded material columns. Each material gets its own starting value (intercept), but all materials share a single slope, so the fitted lines are parallel. This is too simple, because the true slope (`R20 * alpha`) is different for every metal.

**Model 2: Slope per material**
Adds interaction features (`material * temperature`). Each material now gets its own intercept and its own slope, which matches the underlying physics and gives a much better fit.

## **What the Code Does**

1. Generates 200 data points (50 per material) with random temperatures between 0 and 200 °C and Gaussian measurement noise.
2. One-hot encodes the material column.
3. Splits the data into 80% training and 20% testing sets.
4. Trains Model 1 (shared slope) with `LinearRegression`.
5. Builds interaction features and trains Model 2 (slope per material).
6. Compares both models using **R²** and **MAE** on the test set.
7. Plots both models over the data points.

## **Output**

- **Metrics:** R² and MAE for both models, printed to the console.
- **Example predictions:** resistance of copper and iron at 100 °C.
- **Graph 1:** "One shared slope". The lines are parallel and miss the tilt of the data, especially for iron and copper.
- **Graph 2:** "Slope per material". Each line follows its own group of points.

Model 2 should show a higher R² (closer to 1) and a lower MAE than Model 1.

## **Key Takeaway**

When the effect of one variable (temperature) depends on another variable (material), a plain linear model with one-hot columns is not enough. Adding interaction terms lets each category have its own slope and fixes the fit.

## **Notes**

- The data is synthetic and generated with `np.random.seed(42)`, so results are reproducible.
- Logistic regression is not used here because it is for classification. Resistance is a continuous value, so linear regression is the right choice.

# **Predicting Resistance of Unseen Metals with Physics-Based Features**

A machine learning project that predicts the electrical resistance of a metal wire from its **material properties** (resistivity and temperature coefficient) and the **temperature**. The goal is to predict resistance for metals the model has **never seen before**.

## **Overview**

The resistance of a metal rises with temperature, following:

```
R = resistivity * (1 + alpha * (T - 20))
```

- `resistivity` is the base resistivity of the metal at 20 °C (nΩ·m)
- `alpha` is the temperature coefficient (per °C)
- `T` is the temperature in °C

The code generates synthetic measurements for 11 metals (silver, copper, gold, aluminum, tungsten, zinc, nickel, iron, platinum, tin, lead). Each metal is measured at 10 temperatures between 0 and 200 °C, with random noise added. This gives 110 measurements in total. Resistance is reported in mΩ for a wire of fixed geometry.

## **Train/Test Split**

The data is **not** split randomly. Instead, the split is done **by metal**:

- **Test metals:** nickel and platinum (the model never sees them during training)
- **Training metals:** the other 9 metals

This checks whether the model has learned the underlying physics, rather than just memorizing the metals it was trained on.

## **The Two Models**

**Model A: Raw inputs**
A linear regression using `resistivity`, `alpha`, and `temperature` as they are. A linear model can only add these terms together. The real relationship needs them **multiplied** (`resistivity * alpha * temperature`), so this model cannot capture it properly and predicts poorly on new metals.

**Model B: Physics-based feature ("rise")**
Adds one engineered feature:

```
rise = resistivity * alpha * (temperature - 20)
```

The model then uses `resistivity` and `rise`. Since `R = resistivity + rise`, the learned coefficients should both be **close to 1** and the intercept close to 0, which the code prints so you can check. This model fits the data very well, including for unseen metals.

## **What the Code Does**

1. Builds a table of 11 metals with their resistivity and temperature coefficient.
2. Generates noisy resistance measurements at 10 temperatures per metal.
3. Splits the data by metal (nickel and platinum held out for testing).
4. Trains Model A on the raw features and evaluates it with **R²** and **MAE**.
5. Creates the `rise` feature and trains Model B.
6. Prints Model B's learned coefficients and compares them with the expected value of 1.
7. Plots predicted vs actual resistance for both models on the test metals.
8. Defines `predict_resistance()` to predict for any new metal, with examples for titanium and a custom metal.
9. Plots predicted resistance vs temperature for all 11 metals.

## **Difference from the Previous Project**

The previous project ("Resistance vs Temperature: Linear Regression with Interaction Terms") used two models on 4 metals. This project takes a different approach.

| | **Previous project** | **This project** |
|---|---|---|
| **How the metal is described** | By name (one-hot encoded) | By numbers: resistivity and alpha |
| **Number of metals** | 4 | 11 |
| **Train/test split** | Random split of rows | Split by metal (unseen metals in the test set) |
| **Can predict a new metal?** | No, only the 4 metals it was trained on | Yes, any metal if you know its resistivity and alpha |
| **How the model was improved** | Interaction terms (`material * temperature`) | One physics-based feature (`rise`) |
| **Number of learned parameters** | A separate slope and intercept per metal | Just 2 coefficients and an intercept, shared by all metals |
| **Interpretability** | Slopes per metal | Coefficients should be about 1, matching the physics |
| **What it demonstrates** | Interaction terms let each category have its own slope | Feature engineering lets a linear model generalize to new materials |

In short, the previous project learned **a separate line for each metal**. This project learns **one general rule** that works for any metal.

## **Requirements**

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

Install them with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## **Usage**

Run the script, or open the code in Google Colab or Jupyter Notebook and run the cells from top to bottom.

```bash
python your_script_name.py
```

To predict resistance for a new metal:

```python
predict_resistance(resistivity=420, alpha=0.0035, temperature=100)
```

## **Output**

- **Metrics:** R² and MAE for Model A on the test metals.
- **Coefficients:** the values Model B learned (expected to be close to 1).
- **Graph 1:** two side-by-side plots of predicted vs actual resistance for nickel and platinum. Points on the red dashed line are perfect predictions. Model B's points should sit much closer to it.
- **Example predictions:** titanium at 100 °C and a custom metal at 150 °C.
- **Graph 2:** predicted resistance vs temperature for all 11 metals.

## **Key Takeaway**

A plain linear model cannot learn a relationship that depends on a product of inputs. By giving the model the right feature (`rise`), a simple linear regression becomes accurate and can generalize to metals it has never seen. Good features based on domain knowledge often matter more than a more complex model.

## **Notes**

- The data is synthetic and generated with `np.random.seed(0)`, so results are reproducible.
- Predictions for metals far outside the training range (for example, titanium with a resistivity of 420) are extrapolations and should be treated with some caution.
- Logistic regression is not used because it is for classification. Resistance is a continuous value, so linear regression is the right choice.
