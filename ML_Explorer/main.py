import streamlit as st
import pandas as pd
import numpy as np
import traceback
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile
import os
from pandas.errors import EmptyDataError

st.set_page_config(page_title="ML Classification & Regression App (More Models)", layout="wide")

# ---------------------------------------------
# Load built-in dataset
# ---------------------------------------------
@st.cache_data
def load_builtin_dataset(name):
    if name == "Iris":
        data = datasets.load_iris(as_frame=True)
    elif name == "Wine":
        data = datasets.load_wine(as_frame=True)
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer(as_frame=True)
    elif name == "Digits":
        data = datasets.load_digits(as_frame=True)
    else:
        return None
    return data.data, data.target, data.feature_names

# ---------------------------------------------
# Safe CSV/Excel Reader
# ---------------------------------------------
def safe_read(uploaded):
    if uploaded is None:
        return None, "No file uploaded"
    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
        uploaded.seek(0)
        if df.shape[1] == 0:
            return None, "The file has no columns."
        return df, None
    except EmptyDataError:
        return None, "Uploaded file is empty."
    except Exception:
        try:
            uploaded.seek(0)
            df = pd.read_excel(uploaded)
            uploaded.seek(0)
            if df.shape[1] == 0:
                return None, "Excel file has no columns."
            return df, None
        except Exception as e2:
            return None, f"Unable to read file: {str(e2)}"

# ---------------------------------------------
# Evaluate Classification
# ---------------------------------------------
def eval_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "rec": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "cm": confusion_matrix(y_test, y_pred),
        "pred": y_pred
    }

# ---------------------------------------------
# Evaluate Regression
# ---------------------------------------------
def eval_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

# ---------------------------------------------
# Get Model (expanded)
# ---------------------------------------------
def get_model(name, params, mode):
    if mode == "classification":
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, C=params.get("C", 1.0))
        elif name == "KNN":
            return KNeighborsClassifier(n_neighbors=params.get("k", 5))
        elif name == "SVM":
            return SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"), probability=True)
        elif name == "Decision Tree":
            return DecisionTreeClassifier(max_depth=params.get("depth", None))
        elif name == "Random Forest":
            return RandomForestClassifier(n_estimators=params.get("n", 100), max_depth=params.get("depth", None))
        elif name == "Naive Bayes":
            return GaussianNB()
        elif name == "ExtraTrees":
            return ExtraTreesClassifier(n_estimators=params.get("n", 100), max_depth=params.get("depth", None))
        elif name == "Gradient Boosting":
            return GradientBoostingClassifier(n_estimators=params.get("n", 100), learning_rate=params.get("lr", 0.1), max_depth=params.get("depth", 3))
        elif name == "AdaBoost":
            return AdaBoostClassifier(n_estimators=params.get("n", 50), learning_rate=params.get("lr", 1.0))
    else:  # regression
        if name == "Linear Regression":
            return LinearRegression()
        elif name == "Random Forest Regressor":
            return RandomForestRegressor(n_estimators=params.get("n", 100), max_depth=params.get("depth", None))
        elif name == "Decision Tree Regressor":
            return DecisionTreeRegressor(max_depth=params.get("depth", None))
        elif name == "KNN Regressor":
            return KNeighborsRegressor(n_neighbors=params.get("k", 5))
        elif name == "SVR":
            return SVR(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"))
        elif name == "Gradient Boosting Regressor":
            return GradientBoostingRegressor(n_estimators=params.get("n", 100), learning_rate=params.get("lr", 0.1), max_depth=params.get("depth", 3))
        elif name == "ExtraTrees Regressor":
            return ExtraTreesRegressor(n_estimators=params.get("n", 100), max_depth=params.get("depth", None))
        elif name == "AdaBoost Regressor":
            return AdaBoostRegressor(n_estimators=params.get("n", 50), learning_rate=params.get("lr", 1.0))
    return None

# ---------------------------------------------
# PCA Plot
# ---------------------------------------------
def plot_pca(X, y):
    try:
        X_numeric = pd.DataFrame(X).select_dtypes(include=[np.number])
        if X_numeric.shape[1] < 1:
            return None
        pca = PCA(n_components=2)
        pts = pca.fit_transform(X_numeric)
        df = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1], "target": pd.Series(y).astype(str)})
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="target", ax=ax)
        return fig
    except Exception:
        return None

# ---------------------------------------------
# Model Download Helper
# ---------------------------------------------
def get_model_bytes(model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            fname = tmp.name
        joblib.dump(model, fname)
        with open(fname, "rb") as f:
            data = f.read()
        try:
            os.remove(fname)
        except Exception:
            pass
        return data
    except Exception:
        return None

# ---------------------------------------------
# APP UI
# ---------------------------------------------
st.title("📊 ML Classification & Regression App — More Models")
st.write("Upload datasets, choose models, tune hyperparameters, and visualize results.")

st.sidebar.header("Dataset Selection")
source = st.sidebar.radio("Choose dataset source:", ["Built-in", "Upload CSV/Excel"])

X = y = None
feature_names = None

if source == "Built-in":
    name = st.sidebar.selectbox("Choose dataset", ["Iris", "Wine", "Breast Cancer", "Digits"])
    X, y, feature_names = load_builtin_dataset(name)
    st.success(f"Loaded built-in dataset: {name}")
else:
    file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if file:
        df, err = safe_read(file)
        if err:
            st.error(err)
            st.stop()
        st.write("Preview:")
        st.dataframe(df.head())
        target = st.sidebar.selectbox("Select target column", ["Select"] + list(df.columns))
        if target != "Select":
            y = df[target]
            X = df.drop(columns=[target])
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            if y.dtype == "object" or y.dtype.name == "category":
                y = LabelEncoder().fit_transform(y.astype(str))

if X is None or y is None:
    st.warning("Upload a dataset or choose a built-in dataset to continue.")
    st.stop()

# Detect problem type (simple heuristic)
if (not pd.api.types.is_object_dtype(y)) and len(np.unique(y)) > 20:
    mode = "regression"
else:
    mode = "classification"
st.sidebar.write(f"Detected problem: **{mode}**")

# Train/test split
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model selection (expanded)
st.sidebar.header("Model & Hyperparameters")
if mode == "classification":
    model_name = st.sidebar.selectbox("Choose model", [
        "Logistic Regression", "KNN", "SVM",
        "Decision Tree", "Random Forest", "Naive Bayes",
        "ExtraTrees", "Gradient Boosting", "AdaBoost"
    ])
else:
    model_name = st.sidebar.selectbox("Choose model", [
        "Linear Regression", "Decision Tree Regressor", "KNN Regressor",
        "SVR", "Random Forest Regressor", "Gradient Boosting Regressor",
        "ExtraTrees Regressor", "AdaBoost Regressor"
    ])

# Hyperparameters dictionary
params = {}
# common params
if "KNN" in model_name:
    params["k"] = st.sidebar.slider("K / neighbors", 1, 30, 5)
if model_name in ["Logistic Regression", "SVM", "SVR"]:
    params["C"] = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0)
if model_name in ["SVM", "SVR"]:
    params["kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"])
if "Decision Tree" in model_name:
    params["depth"] = st.sidebar.slider("Max depth (0 = None)", 0, 50, 0)
    if params["depth"] == 0:
        params["depth"] = None
if "Random Forest" in model_name or "ExtraTrees" in model_name or "Gradient Boosting" in model_name or "AdaBoost" in model_name:
    params["n"] = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
if "Gradient Boosting" in model_name or "AdaBoost" in model_name:
    params["lr"] = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1)

# Build model
model = get_model(model_name, params, mode)
if model is None:
    st.error("Selected model is not implemented.")
    st.stop()

# Optional scaling (simple)
use_scaler = st.sidebar.checkbox("Use StandardScaler", value=True)
if use_scaler:
    model_pipeline = make_pipeline(StandardScaler(), model)
else:
    model_pipeline = make_pipeline(model)

# Train safely
with st.spinner("Training the model..."):
    try:
        model_pipeline.fit(X_train, y_train)
        st.success("Model trained successfully.")
    except Exception as e:
        st.error("Training failed: " + str(e))
        st.text(traceback.format_exc())
        st.stop()

# Evaluate
st.header("📈 Model Results")
if mode == "classification":
    res = eval_classification(model_pipeline, X_test, y_test)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['acc']:.3f}")
    c2.metric("Precision", f"{res['prec']:.3f}")
    c3.metric("Recall", f"{res['rec']:.3f}")
    c4.metric("F1 Score", f"{res['f1']:.3f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(res["cm"], annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)

    # ROC when binary
    if len(np.unique(y)) == 2:
        try:
            y_score = model_pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc = roc_auc_score(y_test, y_score)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax2.plot([0, 1], [0, 1], linestyle="--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.subheader("ROC Curve")
            st.pyplot(fig2)
        except Exception:
            st.info("ROC not available for this model (needs predict_proba).")
else:
    res = eval_regression(model_pipeline, X_test, y_test)
    d1, d2, d3 = st.columns(3)
    d1.metric("MSE", f"{res['mse']:.3f}")
    d2.metric("MAE", f"{res['mae']:.3f}")
    d3.metric("R²", f"{res['r2']:.3f}")

# PCA
st.subheader("PCA Projection (2D)")
fig_p = plot_pca(X, y)
if fig_p is not None:
    st.pyplot(fig_p)
else:
    st.info("PCA could not be generated for this dataset (no numeric features or too small).")

# Download model
st.subheader("Download Trained Model")
model_bytes = get_model_bytes(model_pipeline)
if model_bytes:
    st.download_button("Download model (.joblib)", data=model_bytes, file_name="trained_model.joblib", mime="application/octet-stream")
else:
    st.error("Could not prepare model for download.")

# Feature importances (if applicable)
try:
    final_estimator = model_pipeline.named_steps[list(model_pipeline.named_steps.keys())[-1]]
    importances = getattr(final_estimator, "feature_importances_", None)
    if importances is not None and hasattr(X, "columns"):
        fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
        st.subheader("Feature importances")
        st.dataframe(fi)
except Exception:
    pass

st.markdown("---")
st.write("Tips: If you get 'Unknown label type: continuous', the target is continuous — switch to regression or bin it into classes.")
st.success("App Ready — try different models and hyperparameters!")





