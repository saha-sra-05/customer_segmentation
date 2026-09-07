# Customer Segmentation using K-Means

An end-to-end customer analytics project that segments customers using demographic and spending information. The project includes exploratory data analysis, feature scaling, K-Means clustering, Elbow Method, Silhouette Score, interactive visualizations, segment profiling, and business recommendations.

## Business Problem

Businesses do not have the same relationship with every customer. Customer segmentation helps marketing teams identify groups with similar characteristics and design more relevant campaigns.

## Objectives

- Explore customer demographics and spending behaviour
- Identify meaningful customer groups using unsupervised learning
- Compare possible cluster counts using the Elbow Method and Silhouette Score
- Visualize customer segments
- Translate clusters into actionable marketing strategies

## Dataset

The included `data/Mall_Customers.csv` is the commonly used Mall Customers dataset with 200 customer records and five columns:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

The dataset is publicly available and widely used for K-Means/customer segmentation examples. Source reference: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## Methodology

1. Load the customer data
2. Validate and clean required fields
3. Perform EDA
4. Select Annual Income and Spending Score for clustering
5. Standardize the features using `StandardScaler`
6. Test K values from 2 to 10
7. Use Elbow Method and Silhouette Score to evaluate K
8. Train K-Means with the K selected in the dashboard
9. Profile each cluster
10. Generate business recommendations

## Streamlit App

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. You can also upload another CSV from the sidebar if it contains the required customer columns.

### Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data` folder.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Create a new app and select your repository.
5. Set the main file to `app.py`.
6. Deploy.

## Project Structure

```text
customer-segmentation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── Mall_Customers.csv
```

## Key Skills Demonstrated

Python, Pandas, NumPy, Scikit-learn, K-Means Clustering, Unsupervised Learning, Feature Scaling, Exploratory Data Analysis, Plotly, Streamlit, Customer Analytics, Data Visualization, Business Insights.

## Important Note

K-Means is an unsupervised learning method, so there is no conventional accuracy score. Cluster quality is assessed using measures such as inertia and silhouette score, together with business interpretability.


OUTPUT :
https://customersegmentation-grjyeahijwdv5n9ytkaj7v.streamlit.app/
