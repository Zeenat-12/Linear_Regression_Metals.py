# **Predicting Metal Resistance vs Temperature with Linear Regression**

A two-part machine learning project that predicts the electrical resistance of metal wires from temperature and material. Both parts use simple `LinearRegression`, and both show how **better features** make a linear model much more accurate.

- **Part 1** shows why **interaction terms** matter (separate slope per material).
- **Part 2** shows how a **physics-based feature** lets the model predict metals it has never seen.

## **Background**

The resistance of a metal rises with temperature, following:

```
R = R20 * (1 + alpha * (T - 20))
```

- `R20` is the resistance (or resistivity) at 20 °C
- `alpha` is the temperature coefficient of the material
- `T` is the temperature in °C

The slope of resistance against temperature is `R20 * alpha`, which is **different for every metal**. This is the key idea behind both parts. All data in this project is synthetic, with random measurement noise added.

## **Project Structure**

| File | Description |
|------|-------------|
| `part1_interaction_terms.py` | Compares a shared-slope model with a slope-per-material model on 4 metals |
| `part2_unseen_metals.py` | Predicts resistance for unseen metals using a physics-based feature |

*(Rename these to match your actual file names.)*

---

# **Part 1: Interaction Terms**

## **Data**

200 measurements (50 per material) with random temperatures between 0 and 200 °C and Gaussian noise (`seed 42`).

| Material | R20 (ohm) | alpha |
|----------|-----------|-------|
| Copper   | 0.168     | 0.0039 |
| Aluminum | 0.282     | 0.0043 |
| Iron     | 0.971     | 0.0065 |
| Tungsten | 0.560     | 0.0045 |

## **The Two Models**

**Model 1: One shared slope**
Uses temperature plus one-hot encoded material columns. Each material gets its own starting value (intercept), but all materials share a single slope, so the fitted lines are parallel. This is too simple, because the true slope is different for every metal.

**Model 2: Slope per material**
Adds interaction features (`material * temperature`). Each material gets its own intercept and its own slope, which matches the physics and gives a much better fit.

## **What the Code Does**

1. Generates the data and one-hot encodes the material column.
2. Splits the data into 80% training and 20% testing sets (random split).
3. Trains Model 1 (shared slope).
4. Builds interaction features and trains Model 2 (slope per material).
5. Compares both models using **R²** and **MAE** on the test set.
6. Plots both models over the data points.

## **Output**

- **Metrics:** R² and MAE for both models.
- **Example predictions:** resistance of copper and iron at 100 °C.
- **Graph 1:** "One shared slope". The lines are parallel and miss the tilt of the data, especially for iron and copper.
- **Graph 2:** "Slope per material". Each line follows its own group of points.

Model 2 should show a higher R² (closer to 1) and a lower MAE than Model 1.

## **Limitation**

Because each metal is represented by its **name**, the model can only predict the 4 metals it was trained on.

---

# **Part 2: Predicting Unseen Metals**

## **Data**

110 measurements: 11 metals (silver, copper, gold, aluminum, tungsten, zinc, nickel, iron, platinum, tin, lead), each measured at 10 temperatures between 0 and 200 °C, with noise (`seed 0`). Each metal is described by two numbers: its **resistivity** (nΩ·m) and its **alpha** (per °C). Resistance is reported in mΩ for a wire of fixed geometry.

## **Train/Test Split**

The split is done **by metal**, not randomly:

- **Test metals:** nickel and platinum (never seen during training)
- **Training metals:** the other 9 metals

This checks whether the model learned the underlying physics instead of memorizing metals.

## **The Two Models**

**Model A: Raw inputs**
A linear regression using `resistivity`, `alpha`, and `temperature` as they are. A linear model can only add these terms, but the real relationship needs them **multiplied** (`resistivity * alpha * temperature`), so this model predicts poorly on new metals.

**Model B: Physics-based feature ("rise")**
Adds one engineered feature:

```
rise = resistivity * alpha * (temperature - 20)
```

The model uses `resistivity` and `rise`. Since `R = resistivity + rise`, both learned coefficients should be **close to 1** and the intercept close to 0, which the code prints so you can check.

## **What the Code Does**

1. Builds a table of 11 metals with resistivity and alpha.
2. Generates noisy resistance measurements at 10 temperatures per metal.
3. Splits the data by metal (nickel and platinum held out).
4. Trains Model A and evaluates it with **R²** and **MAE**.
5. Creates the `rise` feature and trains Model B.
6. Prints Model B's coefficients and compares them with the expected value of 1.
7. Plots predicted vs actual resistance for both models on the test metals.
8. Defines `predict_resistance()` to predict for any new metal (examples: titanium and a custom metal).
9. Plots predicted resistance vs temperature for all 11 metals.

## **Output**

- **Metrics:** R² and MAE for Model A on the test metals.
- **Coefficients:** values learned by Model B (expected to be close to 1).
- **Graph 1:** predicted vs actual resistance for nickel and platinum. Points on the red dashed line are perfect predictions. Model B's points should sit much closer to it.
- **Example predictions:** titanium at 100 °C and a custom metal at 150 °C.
- **Graph 2:** predicted resistance vs temperature for all 11 metals.

To predict a new metal:

```python
predict_resistance(resistivity=420, alpha=0.0035, temperature=100)
```

---

# **Comparing Part 1 and Part 2**

| | **Part 1** | **Part 2** |
|---|---|---|
| **How the metal is described** | By name (one-hot encoded) | By numbers: resistivity and alpha |
| **Number of metals** | 4 | 11 |
| **Train/test split** | Random split of rows | Split by metal (unseen metals in test set) |
| **Can predict a new metal?** | No | Yes, if you know its resistivity and alpha |
| **How the model is improved** | Interaction terms (`material * temperature`) | One physics-based feature (`rise`) |
| **Learned parameters** | A separate slope and intercept per metal | 2 coefficients and an intercept shared by all metals |
| **Interpretability** | Slopes per metal | Coefficients should be about 1, matching the physics |
| **What it demonstrates** | Interaction terms let each category have its own slope | Feature engineering lets a linear model generalize to new materials |

In short, Part 1 learns **a separate line for each metal**, while Part 2 learns **one general rule** that works for any metal.

---

# **Requirements**

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

Install them with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

# **Usage**

Run each script, or open the code in Google Colab or Jupyter Notebook and run the cells from top to bottom.

```bash
python part1_interaction_terms.py
python part2_unseen_metals.py
```

# **Key Takeaways**

- When the effect of one variable (temperature) depends on another (material), a plain linear model is not enough. **Interaction terms** let each category have its own slope.
- A linear model cannot learn a relationship based on a product of inputs. Giving it the right **engineered feature** (`rise`) makes it accurate and lets it generalize to unseen materials.
- Good features based on domain knowledge often matter more than a more complex model.

# **Notes**

- All data is synthetic and generated with fixed random seeds (42 for Part 1, 0 for Part 2), so results are reproducible.
- Predictions for metals far outside the training range (for example, titanium with a resistivity of 420) are extrapolations and should be treated with caution.
- Logistic regression is not used because it is for classification. Resistance is a continuous value, so linear regression is the right choice.
