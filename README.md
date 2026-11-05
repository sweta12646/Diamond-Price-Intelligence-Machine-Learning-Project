
# 💎 Diamond Price Intelligence
### Machine Learning Based Diamond Price Prediction and Classification

## 📌 Project Overview

Diamond Price Intelligence is a Machine Learning project designed to predict the estimated price of diamonds based on their physical measurements and quality characteristics.

This project uses both Regression and Classification algorithms to analyze diamond data and make predictions.

The application is developed using Python, Scikit-learn, Pandas, and Streamlit. It provides an interactive dashboard where users can enter diamond specifications, select machine learning algorithms, predict diamond prices, and compare model performance.

---

## 🎯 Project Objectives

- Predict diamond prices using Regression algorithms.
- Classify diamonds into Budget, Mid-range, and Luxury categories.
- Compare multiple Machine Learning algorithms.
- Evaluate model performance using different metrics.
- Develop an interactive and user-friendly Streamlit application.

---

## 📂 Dataset Information

**Dataset:** Diamonds Dataset

**Source:** Seaborn built-in dataset.

The dataset contains information about diamond characteristics and their corresponding prices.

### Features

| Feature | Description |
|---|---|
| carat | Weight of the diamond |
| cut | Quality of the diamond cut |
| color | Diamond color grade |
| clarity | Diamond clarity grade |
| depth | Total depth percentage |
| table | Width of the top facet percentage |
| x | Length in mm |
| y | Width in mm |
| z | Depth in mm |
| price | Diamond price in USD |

**Target Variable:** `price`

---

## 🤖 Machine Learning Algorithms

### 1. Regression Algorithms

Regression models predict the actual numerical price of a diamond.

| Algorithm | Description |
|---|---|
| Linear Regression | Predicts price using a linear relationship |
| Decision Tree Regressor | Captures nonlinear relationships |
| Random Forest Regressor | Uses multiple decision trees |
| Gradient Boosting Regressor | Combines sequentially improved models |

### 2. Classification Algorithms

Classification models predict the price category of a diamond.

| Algorithm | Description |
|---|---|
| Decision Tree Classifier | Classifies diamonds using decision rules |
| Random Forest Classifier | Uses multiple decision trees for classification |
| Logistic Regression | Predicts the price category |

### Classification Categories

- **Budget:** Lower-price diamonds.
- **Mid-range:** Medium-price diamonds.
- **Luxury:** Higher-price diamonds.

The categories are created using the training dataset's price tertiles.

---

## 📊 Model Evaluation Metrics

### Regression Metrics

| Metric | Description | Ideal Value |
|---|---|---|
| R² Score | Measures how well the model explains price variation | Higher |
| MAE | Measures average absolute prediction error | Lower |
| RMSE | Measures prediction error with greater penalty for large errors | Lower |

### Classification Metric

| Metric | Description | Ideal Value |
|---|---|---|
| Accuracy | Measures the percentage of correct predictions | Higher |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

```text
Diamond_Prediction/
│
├── train_model.py
│
├── app.py
│
├── diamond_models.pkl
│
└── README.md
```

---

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Diamond_Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Diamond_Prediction
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy seaborn scikit-learn streamlit joblib matplotlib
```

### Step 4: Train the Machine Learning Models

Run the training script:

```bash
python train_model.py
```

This script will:

- Load the Diamonds dataset.
- Clean and preprocess the data.
- Train four regression models.
- Train three classification models.
- Compare model performance.
- Identify the best-performing models.
- Save all trained models into `diamond_models.pkl`.

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 🌟 Application Features

### 1. Price Prediction

- Enter diamond measurements.
- Select cut, color, and clarity.
- Choose a regression algorithm.
- Predict the estimated diamond price.

### 2. Price Classification

- Choose a classification algorithm.
- Predict whether the diamond belongs to Budget, Mid-range, or Luxury category.

### 3. Model Comparison Dashboard

- Compare regression algorithms.
- Compare classification algorithms.
- View R² Score, MAE, RMSE, and Accuracy.
- Identify the best-performing model.

### 4. User-Friendly Interface

- Interactive Streamlit dashboard.
- Sidebar navigation.
- Organized input sections.
- Prediction results.
- Graphical model performance comparison.

---

## 📈 Expected Output

The application provides:

1. Predicted diamond price in USD.
2. Predicted diamond price category.
3. Selected regression and classification algorithms.
4. Regression model comparison table.
5. Classification accuracy comparison.
6. Performance visualization charts.

*Note: Actual model performance depends on the training and testing data.*

---

## 🔮 Future Enhancements

- Add XGBoost and Support Vector Regression.
- Include additional evaluation metrics.
- Add interactive data visualizations.
- Improve prediction accuracy through hyperparameter tuning.
- Deploy the application using Streamlit Community Cloud.
- Add downloadable prediction reports.
- Include actual-versus-predicted price visualizations.

---

## 👩‍💻 Author

**Project Name:** Diamond Price Intelligence

**Domain:** Machine Learning

**Project Type:** Regression and Classification

**Framework:** Streamlit

---

## 📜 License

This project is created for educational and learning purposes.

You are welcome to modify and improve this project for academic and personal use.
