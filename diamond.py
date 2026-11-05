
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Diamond Price Intelligence",
    page_icon="💎",
    layout="wide"
)

# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource
def load_models():
    return joblib.load("diamond_models.pkl")

data = load_models()

reg_models = data["regression_models"]
class_models = data["classification_models"]

reg_results = data["regression_results"]
class_results = data["classification_results"]

best_reg = data["best_regression"]
best_class = data["best_classification"]

# -------------------------------
# CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #172554;
}
.subtitle {
    color: #64748b;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<div class="main-title">💎 Diamond Price Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Diamond Valuation & Analysis</div>',
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("💎 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    ["Price Prediction", "Model Comparison", "About Project"]
)

st.sidebar.divider()

st.sidebar.info(
    "This application uses machine learning "
    "to estimate diamond prices and classify "
    "diamonds into price categories."
)

# -------------------------------
# PRICE PREDICTION
# -------------------------------

if page == "Price Prediction":

    st.subheader("🔍 Enter Diamond Details")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Physical Measurements")

        carat = st.slider(
            "Carat Weight",
            0.2, 5.0, 0.7, 0.01
        )

        depth = st.slider(
            "Depth Percentage",
            43.0, 79.0, 61.7, 0.1
        )

        table = st.slider(
            "Table Percentage",
            43.0, 79.0, 57.4, 0.1
        )

        x = st.slider(
            "Length (mm)",
            3.0, 11.0, 5.5, 0.01
        )

        y = st.slider(
            "Width (mm)",
            3.0, 11.0, 5.5, 0.01
        )

        z = st.slider(
            "Depth (mm)",
            1.0, 7.0, 3.4, 0.01
        )

    with col2:

        st.markdown("### Diamond Quality")

        cut = st.selectbox(
            "Cut Quality",
            ["Fair", "Good", "Very Good",
             "Premium", "Ideal"],
            index=4
        )

        color = st.selectbox(
            "Diamond Color",
            ["D", "E", "F", "G", "H", "I", "J"],
            index=3
        )

        clarity = st.selectbox(
            "Diamond Clarity",
            ["I1", "SI2", "SI1", "VS2",
             "VS1", "VVS2", "VVS1", "IF"],
            index=3
        )

        st.markdown("### Select Algorithm")

        regression_choice = st.selectbox(
            "Price Prediction Model",
            list(reg_models.keys()),
            index=list(reg_models.keys()).index(best_reg)
        )

        classification_choice = st.selectbox(
            "Price Category Model",
            list(class_models.keys()),
            index=list(class_models.keys()).index(best_class)
        )

    st.divider()

    # Input dataframe
    input_data = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    if st.button("💎 Predict Diamond Value", use_container_width=True):

        # Regression prediction
        predicted_price = reg_models[
            regression_choice
        ].predict(input_data)[0]

        # Classification prediction
        predicted_category = class_models[
            classification_choice
        ].predict(input_data)[0]

        st.success("Prediction completed successfully!")

        st.markdown("### 💰 Estimated Diamond Value")

        st.metric(
            "Predicted Price",
            f"${predicted_price:,.2f}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"**Price Category:** {predicted_category}"
            )

        with col2:
            st.info(
                f"**Regression Model:** {regression_choice}"
            )

        st.caption(
            "This is a machine learning estimate based on "
            "the training dataset, not a professional appraisal."
        )

# -------------------------------
# MODEL COMPARISON
# -------------------------------

elif page == "Model Comparison":

    st.subheader("📊 Machine Learning Model Comparison")

    st.markdown("### Regression Performance")

    regression_df = pd.DataFrame(
        reg_results
    ).T.reset_index()

    regression_df.columns = [
        "Model", "R2 Score", "MAE", "RMSE"
    ]

    st.dataframe(
        regression_df.style.format({
            "R2 Score": "{:.4f}",
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### R² Score Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        regression_df["Model"],
        regression_df["R2 Score"]
    )

    ax.set_ylabel("R² Score")
    ax.set_xlabel("Regression Models")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    st.success(
        f"Best Regression Model: {best_reg}"
    )

    st.divider()

    st.markdown("### Classification Performance")

    classification_df = pd.DataFrame(
        list(class_results.items()),
        columns=["Model", "Accuracy"]
    )

    st.dataframe(
        classification_df.style.format({
            "Accuracy": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        classification_df["Model"],
        classification_df["Accuracy"]
    )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classification Models")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    st.success(
        f"Best Classification Model: {best_class}"
    )

# -------------------------------
# ABOUT PROJECT
# -------------------------------

elif page == "About Project":

    st.subheader("📘 About This Project")

    st.write("""
    This project uses the Seaborn Diamonds dataset
    to build machine learning models.

    **Regression:**
    Predicts the estimated price of a diamond.

    **Classification:**
    Classifies diamonds into Budget, Mid-range,
    and Luxury price categories.

    **Algorithms used:**

    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Decision Tree Classifier
    - Random Forest Classifier
    - Logistic Regression

    **Evaluation metrics:**

    - R² Score
    - Mean Absolute Error
    - Root Mean Squared Error
    - Classification Accuracy
    """)

    st.markdown("### Dataset Information")

    df = sns.load_dataset("diamonds")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Features", len(df.columns) - 1)
    col3.metric("Target Variable", "Price")

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

st.divider()

st.caption(
    "Diamond Price Intelligence | Machine Learning Project"
)
