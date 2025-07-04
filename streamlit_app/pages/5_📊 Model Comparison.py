import streamlit as st
import pandas as pd

st.set_page_config(page_title="📊 Model Comparison",page_icon="📊")
st.title("📈 Model Comparison")

model_metrics = [
    {
        'Model': 'Logistic Regression',
        'Accuracy': 0.8150,
        'ROC AUC': 0.7801,
        'Class 1 Precision': 0.58,
        'Class 1 Recall': 0.21,
        'Class 1 F1-score': 0.31
    },
    {
        'Model': 'Random Forest (Tuned)',
        'Accuracy': 0.844,
        'ROC AUC': 0.8698,
        'Class 1 Precision': 0.61,
        'Class 1 Recall': 0.67,
        'Class 1 F1-score': 0.64
    },
    {
        'Model': 'XGBoost (Tuned)',
        'Accuracy': 0.829,
        'ROC AUC': 0.8758,
        'Class 1 Precision': 0.56,
        'Class 1 Recall': 0.71,
        'Class 1 F1-score': 0.63
    },
    {
        'Model': 'Stacked Ensemble (all 3)',
        'Accuracy': 0.835,
        'ROC AUC': 0.8646,
        'Class 1 Precision': 0.58,
        'Class 1 Recall': 0.73,
        'Class 1 F1-score': 0.64
    }
]

df = pd.DataFrame(model_metrics)

st.markdown("### 🔍 Performance Comparison Table")
st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))

st.markdown("### 📊 Compare Models by Metric")

metric_options = ["Accuracy", "ROC AUC", "Class 1 Precision", "Class 1 Recall", "Class 1 F1-score"]
selected_metric = st.selectbox("Select a metric to visualize:", metric_options)

st.bar_chart(df.set_index("Model")[selected_metric])

st.markdown("### 📈 Metric Trends Across Models")

df_long = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
line_data = df_long.pivot(index="Metric", columns="Model", values="Score")

metric_order = ["Accuracy", "ROC AUC", "Class 1 Precision", "Class 1 Recall", "Class 1 F1-score"]
line_data = line_data.reindex(metric_order)

st.line_chart(line_data)
