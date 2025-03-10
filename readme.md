# Customer Churn Prediction & Retention Strategy Optimization

🚀 **AI-Powered Predictive Analytics for Customer Retention**

## 📌 Project Overview

This project leverages machine learning (XGBoost, PyCaret), NLP (Sentence Transformers), and RAG (LangChain + Mistral) to predict customer churn and provide personalized retention strategies. The goal is to analyze customer behavior, contract types, payment methods, and churn reasons to identify high-risk customers and recommend tailored retention actions.

## 🎯 Key Objectives

✅ Develop a predictive churn model using historical customer data.
✅ Identify key factors influencing churn (e.g., contract type, monthly charges, tenure).
✅ Convert customer notes into embeddings for text-based churn insights.
✅ Store & retrieve past churn cases using ChromaDB for case-based learning.
✅ Use Retrieval-Augmented Generation (RAG) with LangChain + Mistral to explain predictions.
✅ Generate personalized customer retention strategies based on insights.

## 📊 Datasets Used

📌 **Churn Prediction Dataset** – Kaggle
📌 **Telco Customer Churn Dataset** – Kaggle

## 🛠️ Tech Stack

- **ML Models:** XGBoost, PyCaret, Scikit-Learn
- **NLP & Embeddings:** Sentence Transformers, LangChain
- **Vector Database:** ChromaDB (for retrieving similar past churn cases)
- **Data Processing:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn, Power BI
- **LLM & RAG:** Mistral, LangChain
- **Deployment & Dashboarding:** Power BI, Streamlit

## 📊 Exploratory Data Analysis (EDA) Insights

✔️ 26.5% of customers churn – highlighting a class imbalance challenge.\
✔️ Month-to-month contracts show the highest churn rate – long-term contracts reduce churn risk.\
✔️ Senior citizens & high monthly charges correlate with higher churn – indicating possible financial constraints.\
✔️ Missing values in churn reasons & outliers in revenue detected – requiring preprocessing.

## 🚀 Next Steps

📍 Implement data preprocessing to handle missing values and outliers.\
📍 Train the XGBoost churn prediction model and evaluate performance.\
📍 Develop the RAG-based system for AI-driven churn explanations.\
📍 Build a Power BI dashboard for insights visualization.

## 📌 How to Run the Project

1️⃣ **Clone the repository:**

```bash
git clone https://github.com/your-username/Churn-Forecasting-and-Strategic-Retention-Using-Data-Analytics---A.git
cd Churn-Forecasting-and-Strategic-Retention-Using-Data-Analytics---A
```

2️⃣ **Install dependencies:**

```bash
pip install -r requirements.txt
```

3️⃣ **Run data preprocessing:**

```bash
python src/data_processing.py
```

4️⃣ **Train the model:**

```bash
python src/train_model.py
```

5️⃣ **Run the RAG explainer:**

```bash
python src/rag_explainer.py
```

6️⃣ **Launch the Streamlit dashboard:**

```bash
streamlit run dashboards/Streamlit/app.py
```

## 📜 License

This project is licensed under the Apache 2.0 License.

---

This README ensures stakeholders quickly understand the project scope, objectives, and implementation steps. Let me know if you need further enhancements! 🚀


