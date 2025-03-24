# Unlocking YouTube Channel Performance Secrets

## 📌 Project Overview
This project focuses on analyzing YouTube channel performance using **Machine Learning (ML), Financial Analysis (FA), and Data Analytics (DA)**. The goal is to uncover patterns and trends that affect **video engagement, monetization, and audience retention**.

### **🔧 Tools & Technologies Used**
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
- **Tools**: Jupyter Notebook, Visual Studio Code
- **Dataset**: YouTube video analytics dataset (Downloadable via the provided link)

---

## 📂 Dataset Description
The dataset contains key performance metrics of YouTube videos, including:

### **🎥 Video Details**
- **Video Duration**
- **Publish Time**
- **Days Since Published**
- **Day of the Week**

### **💰 Revenue Metrics**
- **Revenue per 1000 Views (USD)**
- **Estimated Revenue (USD)**
- **Ad Impressions**
- **Playback-Based CPM**
- **YouTube Premium Revenue**

### **📊 Engagement Metrics**
- **Views, Likes, Dislikes, Shares, Comments**
- **Watch Time (Hours)**
- **Average View Duration**
- **Video Thumbnail Click-Through Rate (CTR)**

### **👥 Audience Insights**
- **New Subscribers, Unsubscribes**
- **Unique Viewers, Returning Viewers**

---

## 📊 Step-by-Step Analysis & Machine Learning Workflow

### **1️⃣ Data Preprocessing & Cleaning**
```python
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("youtube_channel_data.csv")

# Check for missing values
data.isnull().sum()

# Convert 'Video Duration' into seconds
import isodate
data['Video Duration'] = data['Video Duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())

# Drop missing values
data.dropna(inplace=True)
```

### **2️⃣ Exploratory Data Analysis (EDA)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing relationships
sns.pairplot(data[['Revenue per 1000 Views (USD)', 'Views', 'Subscribers', 'Estimated Revenue (USD)']])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

### **3️⃣ Feature Engineering**
```python
# Creating new features
data['Revenue per View'] = data['Estimated Revenue (USD)'] / data['Views']
data['Engagement Rate'] = (data['Likes'] + data['Shares'] + data['Comments']) / data['Views'] * 100
```

### **4️⃣ Data Visualization**
```python
# Revenue Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Revenue (USD)'], bins=50, kde=True, color='green')
plt.title("Revenue Distribution")
plt.xlabel("Revenue (USD)")
plt.ylabel("Frequency")
plt.show()

# Revenue vs Views
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Views'], y=data['Estimated Revenue (USD)'], alpha=0.7)
plt.title("Revenue vs Views")
plt.xlabel("Views")
plt.ylabel("Revenue (USD)")
plt.show()
```

### **5️⃣ Predictive Modeling: Revenue Estimation**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable
features = ['Views', 'Subscribers', 'Likes', 'Shares', 'Comments', 'Engagement Rate']
target = 'Estimated Revenue (USD)'
X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
```

### **6️⃣ Feature Importance Analysis**
```python
# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance")
plt.show()
```

### **7️⃣ Model Deployment**
```python
import joblib
joblib.dump(model, 'youtube_revenue_predictor.pkl')
```

---

## 📌 Key Findings & Insights
✅ **Engagement Metrics (likes, shares, comments) significantly impact revenue.**  
✅ **Views alone don’t determine revenue—monetization strategy matters.**  
✅ **Publishing time and video duration affect audience retention.**  
✅ **Machine Learning models can help predict revenue and optimize content strategies.**  
✅ **Data-driven decision-making helps YouTube creators grow their channels effectively.**  

---

## 🎯 Conclusion
This project provides actionable insights into **YouTube channel performance** by leveraging **data analytics and machine learning**. It helps content creators and marketers optimize their strategies for **better engagement, higher ad revenue, and sustainable growth**. 🚀

---

## 📌 How to Use This Repository
1. Clone the repository:  
   ```sh
   git clone https://github.com/YourUsername/YourRepository.git
   ```
2. Install dependencies:  
   ```sh
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```
3. Run the Jupyter Notebook or Python scripts to analyze and predict YouTube revenue.

Feel free to contribute, raise issues, or fork the repository! 😊
