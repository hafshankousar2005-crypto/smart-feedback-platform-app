import streamlit as st
st.set_page_config(page_title="Smart Feedback Intelligence Platform", layout="wide")
st.title("📊 Smart Feedback Intelligence Platform")
st.markdown("---")

# ================================================================================
# SECTION 1: IMPORTS (All original imports)
# ================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import os
from collections import Counter
from textblob import TextBlob
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# CACHING – ensures data generation and model training run only once
# (this does NOT remove any logic, only speeds up the app)
# --------------------------------------------------------------------------------
@st.cache_data
def generate_data():
    # ----- Your original data generation (copied exactly) -----
    customers = [
        {"id": "CUST1001", "name": "John Smith", "segment": "Premium", "tenure": 24},
        {"id": "CUST1002", "name": "Sarah Johnson", "segment": "Regular", "tenure": 12},
        {"id": "CUST1003", "name": "Mike Chen", "segment": "Premium", "tenure": 36},
        {"id": "CUST1004", "name": "Emma Wilson", "segment": "New", "tenure": 2},
        {"id": "CUST1005", "name": "David Brown", "segment": "Regular", "tenure": 18},
        {"id": "CUST1006", "name": "Lisa Anderson", "segment": "Premium", "tenure": 48},
        {"id": "CUST1007", "name": "James Taylor", "segment": "New", "tenure": 1},
        {"id": "CUST1008", "name": "Maria Garcia", "segment": "Regular", "tenure": 8},
        {"id": "CUST1009", "name": "Robert Lee", "segment": "Premium", "tenure": 60},
        {"id": "CUST1010", "name": "Patricia White", "segment": "Regular", "tenure": 15}
    ]

    products = [
        {"id": "PROD101", "name": "SmartPhone X12", "category": "Electronics", "price": 699},
        {"id": "PROD102", "name": "Laptop Pro", "category": "Electronics", "price": 1299},
        {"id": "PROD103", "name": "Wireless Earbuds", "category": "Electronics", "price": 149},
        {"id": "PROD104", "name": "Coffee Maker", "category": "Home", "price": 89},
        {"id": "PROD105", "name": "Running Shoes", "category": "Sports", "price": 129},
        {"id": "PROD106", "name": "Desk Chair", "category": "Furniture", "price": 199},
        {"id": "PROD107", "name": "LED Lamp", "category": "Home", "price": 45},
        {"id": "PROD108", "name": "Power Bank", "category": "Electronics", "price": 39}
    ]

    locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]
    channels = ["Mobile App", "Website", "In-Store", "Social Media", "Email"]
    categories = ["Product Quality", "Delivery", "Customer Service", "Website Experience", 
                  "Price", "Returns", "Packaging", "Product Features"]

    data = []
    for i in range(200):
        customer = random.choice(customers)
        product = random.choice(products)
        location = random.choice(locations)
        channel = random.choice(channels)
        category = random.choice(categories)
        days_ago = random.randint(0, 90)
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
        rating = random.choices([5,4,3,2,1], weights=[40,25,15,12,8])[0]
        if rating >= 4:
            sentiment = "Positive"
            texts = [
                f"Great {product['name']}! Really love the quality.",
                f"Excellent service! Very happy with my purchase.",
                f"The {product['name']} exceeded my expectations.",
                f"Fast delivery and perfect packaging. Thank you!",
                f"Amazing product! Will definitely buy again.",
                f"Best {product['category']} product I've ever bought.",
                f"Customer service was very helpful and friendly.",
                f"Great value for money. Highly recommended!"
            ]
        elif rating == 3:
            sentiment = "Neutral"
            texts = [
                f"The {product['name']} is okay, nothing special.",
                f"Average product. Does the job.",
                f"Delivery was on time. Product works fine.",
                f"It's decent for the price.",
                f"Not bad, but could be better."
            ]
        else:
            sentiment = "Negative"
            texts = [
                f"Very disappointed with {product['name']}. Poor quality.",
                f"Terrible customer service. No one responded.",
                f"The product stopped working after a few days.",
                f"Delivery was very late. Package was damaged.",
                f"Not worth the money. Cheap quality.",
                f"Website kept crashing during checkout.",
                f"Received wrong item. Return process is complicated."
            ]
        feedback_text = random.choice(texts)
        order_value = product['price'] * random.uniform(0.8, 1.2)
        returned = random.choices([True, False], weights=[15, 85])[0] if rating <= 2 else False
        response_time = random.randint(5, 120) if rating >= 4 else random.randint(30, 1440)
        resolved = random.choices([True, False], weights=[90, 10])[0] if rating >= 3 else random.choices([True, False], weights=[70, 30])[0]

        data.append({
            'feedback_id': f"FBK{str(i+1).zfill(4)}",
            'timestamp': timestamp,
            'customer_id': customer['id'],
            'customer_name': customer['name'],
            'customer_segment': customer['segment'],
            'customer_tenure': customer['tenure'],
            'product_name': product['name'],
            'product_category': product['category'],
            'feedback_text': feedback_text,
            'rating': rating,
            'sentiment': sentiment,
            'category': category,
            'channel': channel,
            'location': location,
            'order_value': round(order_value, 2),
            'returned': returned,
            'response_time_mins': response_time,
            'resolved': resolved
        })

    df = pd.DataFrame(data)

    # Deep sentiment analysis (original function)
    def analyze_sentiment_deep(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        words = text.lower().split()
        positive_words = sum(1 for w in words if w in ['great', 'excellent', 'amazing', 'love', 'best', 'good', 'happy'])
        negative_words = sum(1 for w in words if w in ['bad', 'worst', 'terrible', 'poor', 'disappointed', 'late', 'damaged'])
        composite = (polarity + 1) * 50
        composite = max(0, min(100, composite))
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'composite_score': composite,
            'positive_words': positive_words,
            'negative_words': negative_words
        }

    sentiment_results = df['feedback_text'].apply(analyze_sentiment_deep)
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    df = pd.concat([df, sentiment_df], axis=1)
    return df

@st.cache_resource
def train_models(df):
    # ----- Your original ML preparation and training (copied exactly) -----
    df['churn_risk'] = ((df['rating'] <= 2) | (df['returned'] == True)).astype(int)
    df['satisfaction_level'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], labels=['Low', 'Medium', 'High'])

    le_segment = LabelEncoder()
    le_category = LabelEncoder()
    le_channel = LabelEncoder()
    le_location = LabelEncoder()
    df['segment_encoded'] = le_segment.fit_transform(df['customer_segment'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['channel_encoded'] = le_channel.fit_transform(df['channel'])
    df['location_encoded'] = le_location.fit_transform(df['location'])

    feature_columns = [
        'rating', 'order_value', 'response_time_mins', 'customer_tenure',
        'polarity', 'subjectivity', 'composite_score',
        'positive_words', 'negative_words',
        'segment_encoded', 'category_encoded', 'channel_encoded', 'location_encoded'
    ]
    X = df[feature_columns]
    y_churn = df['churn_risk']
    y_satisfaction = df['satisfaction_level']
    y_score = df['composite_score']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

    # Model 1: Churn (3 algorithms)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_churn, test_size=0.25, random_state=42, stratify=y_churn)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_scaled, y_churn, cv=5)
        results[name] = {'accuracy': acc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name
    # Feature importance
    if best_name in ['Random Forest', 'Gradient Boosting']:
        importance = pd.DataFrame({'feature': feature_columns, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
    else:
        coef = best_model.coef_[0]
        importance = pd.DataFrame({'feature': feature_columns, 'importance': np.abs(coef)}).sort_values('importance', ascending=False)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Model 2: Satisfaction multi-class
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_scaled, y_satisfaction, test_size=0.25, random_state=42, stratify=y_satisfaction)
    rf_multi = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_multi.fit(X_train_m, y_train_m)
    accuracy_multi = rf_multi.score(X_test_m, y_test_m)
    class_report = classification_report(y_test_m, rf_multi.predict(X_test_m), output_dict=True)

    # Model 3: Sentiment regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_score, test_size=0.25, random_state=42)
    gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_reg.fit(X_train_r, y_train_r)
    y_pred_r = gb_reg.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)
    imp_reg = pd.DataFrame({'feature': feature_columns, 'importance': gb_reg.feature_importances_}).sort_values('importance', ascending=False)

    return {
        'churn_best_model': best_model,
        'churn_best_name': best_name,
        'churn_accuracy': best_accuracy,
        'churn_cv_results': results,
        'churn_feature_importance': importance,
        'churn_confusion_matrix': cm,
        'sat_model': rf_multi,
        'sat_accuracy': accuracy_multi,
        'sat_class_report': class_report,
        'sentiment_regressor': gb_reg,
        'sentiment_mse': mse,
        'sentiment_mae': mae,
        'sentiment_r2': r2,
        'sentiment_feature_importance': imp_reg,
        'scaler': scaler,
        'label_encoders': {'segment': le_segment, 'category': le_category, 'channel': le_channel, 'location': le_location},
        'feature_columns': feature_columns
    }

# --------------------------------------------------------------------------------
# Generate data and train models (cached – runs once)
# --------------------------------------------------------------------------------
df = generate_data()
df['satisfaction_level'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], labels=['Low', 'Medium', 'High'])
ml = train_models(df)

# ================================================================================
# DISPLAY ORIGINAL OUTPUTS (using Streamlit instead of print/plt.show)
# ================================================================================
st.write("=" * 60)
st.write(" SMART FEEDBACK INTELLIGENCE PLATFORM")
st.write("=" * 60)

st.write("\n Generating realistic feedback data...")
st.write(f" Generated {len(df)} feedback records")
st.write("\n Running sentiment analysis...")
st.write(" Sentiment analysis complete")

# --------------------------------------------------------------------------------
# KPIs and metrics (original prints turned into st.write)
# --------------------------------------------------------------------------------
st.write("\n Calculating business metrics...")
kpis = {
    'Total Feedback': len(df),
    'Average Rating': round(df['rating'].mean(), 2),
    'Positive Rate': f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%",
    'Negative Rate': f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%",
    'Return Rate': f"{df['returned'].mean() * 100:.1f}%",
    'Resolution Rate': f"{df['resolved'].mean() * 100:.1f}%",
    'Avg Response Time': f"{df['response_time_mins'].mean():.0f} mins",
    'Avg Order Value': f"${df['order_value'].mean():.2f}"
}
st.write("\n KEY METRICS:")
for key, value in kpis.items():
    st.write(f"  {key}: {value}")

# Category performance
st.write("\n CATEGORY PERFORMANCE:")
category_perf = df.groupby('category').agg({'rating': 'mean', 'feedback_id': 'count'}).round(2).sort_values('rating', ascending=False)
st.dataframe(category_perf)

# Channel performance
st.write("\n CHANNEL PERFORMANCE:")
channel_perf = df.groupby('channel').agg({'rating': 'mean', 'feedback_id': 'count'}).round(2).sort_values('rating', ascending=False)
st.dataframe(channel_perf)

# Customer segment analysis
st.write("\n CUSTOMER SEGMENT ANALYSIS:")
segment_perf = df.groupby('customer_segment').agg({'rating': 'mean', 'customer_id': 'count', 'order_value': 'mean'}).round(2)
st.dataframe(segment_perf)

# --------------------------------------------------------------------------------
# Insights generation (original logic)
# --------------------------------------------------------------------------------
st.write("\n KEY INSIGHTS:")
insights = []
sentiment_by_day = df.groupby(df['timestamp'].dt.date)['rating'].mean()
if len(sentiment_by_day) > 1:
    trend = sentiment_by_day.iloc[-1] - sentiment_by_day.iloc[0]
    if trend > 0.2:
        insights.append(" Customer satisfaction is improving!")
    elif trend < -0.2:
        insights.append(" Customer satisfaction is declining - take action!")
best_category = category_perf.index[0]
worst_category = category_perf.index[-1]
insights.append(f" Best performing category: {best_category}")
insights.append(f" Needs improvement: {worst_category}")
best_channel = channel_perf.index[0]
worst_channel = channel_perf.index[-1]
insights.append(f" Best channel: {best_channel}")
insights.append(f" Worst channel: {worst_channel}")
if df['returned'].mean() > 0.15:
    insights.append(" High return rate - investigate quality issues")
avg_response = df['response_time_mins'].mean()
if avg_response > 120:
    insights.append(f" Response time too high ({avg_response:.0f} mins)")
for i, insight in enumerate(insights, 1):
    st.write(f"  {i}. {insight}")

# --------------------------------------------------------------------------------
# Recommendations (original logic)
# --------------------------------------------------------------------------------
st.write("\n RECOMMENDATIONS:")
recommendations = []
negative_count = (df['sentiment'] == 'Negative').sum()
if negative_count > 0:
    top_negative_cats = df[df['sentiment'] == 'Negative']['category'].value_counts().head(2)
    recommendations.append(f"Priority 1: Improve {top_negative_cats.index[0]}")
    if len(top_negative_cats) > 1:
        recommendations.append(f"Priority 2: Address {top_negative_cats.index[1]} issues")
return_products = df[df['returned']]['product_name'].value_counts().head(2)
if len(return_products) > 0:
    recommendations.append(f"Review quality of: {', '.join(return_products.index)}")
recommendations.append("Implement customer feedback loop")
recommendations.append("Monitor sentiment trends weekly")
recommendations.append("Train staff on top complaint areas")
recommendations.append("Consider loyalty program for Premium customers")
for i, rec in enumerate(recommendations, 1):
    st.write(f"  {i}. {rec}")

# --------------------------------------------------------------------------------
# Visualizations (original 6 charts)
# --------------------------------------------------------------------------------
st.write("\n Generating visualizations...")
fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 3, 1)
rating_counts = df['rating'].value_counts().sort_index()
bars1 = ax1.bar(rating_counts.index, rating_counts.values, color='skyblue')
ax1.set_title('Rating Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Count')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')

ax2 = plt.subplot(2, 3, 2)
sentiment_counts = df['sentiment'].value_counts()
colors = ['green' if x=='Positive' else 'red' if x=='Negative' else 'gray' for x in sentiment_counts.index]
bars2 = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
ax2.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Count')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')

ax3 = plt.subplot(2, 3, 3)
category_avg = df.groupby('category')['rating'].mean().sort_values()
bars3 = ax3.barh(category_avg.index, category_avg.values, color='lightcoral')
ax3.set_title('Average Rating by Category', fontsize=14, fontweight='bold')
ax3.set_xlabel('Average Rating')
for bar in bars3:
    width = bar.get_width()
    ax3.text(width + 0.05, bar.get_y() + bar.get_height()/2., f'{width:.2f}', ha='left', va='center', fontweight='bold')

ax4 = plt.subplot(2, 3, 4)
channel_counts = df['channel'].value_counts()
ax4.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff99cc'])
ax4.set_title('Feedback by Channel', fontsize=14, fontweight='bold')

ax5 = plt.subplot(2, 3, 5)
segment_avg = df.groupby('customer_segment')['rating'].mean()
bars5 = ax5.bar(segment_avg.index, segment_avg.values, color=['gold', 'silver', 'brown'])
ax5.set_title('Rating by Customer Segment', fontsize=14, fontweight='bold')
ax5.set_xlabel('Segment')
ax5.set_ylabel('Avg Rating')
ax5.set_ylim(0, 5)
for i, (idx, val) in enumerate(segment_avg.items()):
    ax5.text(i, val + 0.1, f'{val:.2f}', ha='center', fontweight='bold')

ax6 = plt.subplot(2, 3, 6)
daily_avg = df.groupby(df['timestamp'].dt.date)['rating'].mean()
ax6.plot(range(len(daily_avg)), daily_avg.values, marker='o', linestyle='-', color='purple', linewidth=2)
ax6.set_title('Rating Trend Over Time', fontsize=14, fontweight='bold')
ax6.set_xlabel('Days (Most Recent First)')
ax6.set_ylabel('Avg Rating')
ax6.grid(True, alpha=0.3)

plt.suptitle('SMART FEEDBACK ANALYZER - COMPLETE DASHBOARD', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig)

# --------------------------------------------------------------------------------
# Export results (original CSV + JSON)
# --------------------------------------------------------------------------------
st.write("\n Exporting results...")
df.to_csv('feedback_analysis_complete.csv', index=False)
st.write(" Data saved to 'feedback_analysis_complete.csv'")
report = {
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_records': len(df),
    'kpi_summary': kpis,
    'top_insights': insights[:5],
    'key_recommendations': recommendations[:5]
}
with open('analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
st.write(" Report saved to 'analysis_report.json'")

st.write("\n" + "=" * 60)
st.write(" ANALYSIS COMPLETE!")
st.write("=" * 60)
st.write("\n Files created:")
st.write("    feedback_analysis_complete.csv - Complete dataset")
st.write("    analysis_report.json - Summary report")
st.write("    Charts displayed above")
st.write("\n Key Findings:")
st.write(f"   Overall Rating: {kpis['Average Rating']}/5.0")
st.write(f"   Positive Feedback: {kpis['Positive Rate']}")
st.write(f"   Top Category: {category_perf.index[0]}")
st.write(f"   Best Channel: {channel_perf.index[0]}")
st.write("=" * 60)

# ================================================================================
# MACHINE LEARNING MODELS (original ML code fully preserved)
# ================================================================================
st.write("\n" + "=" * 60)
st.write(" MACHINE LEARNING MODELS")
st.write("=" * 60)

st.write("\n Preparing data for ML models...")
st.write("\n" + "-" * 40)
st.write(" MODEL 1: Customer Churn Prediction")
st.write("-" * 40)

# Display model comparison results
for name, res in ml['churn_cv_results'].items():
    st.write(f"\n {name}:")
    st.write(f"   Accuracy: {res['accuracy']:.3f}")
    st.write(f"   Cross-validation: {res['cv_mean']:.3f} (+/- {res['cv_std']*2:.3f})")

st.write(f"\n Top 5 Features for Churn Prediction:")
st.dataframe(ml['churn_feature_importance'].head())
st.write(f"\n Best Model: {ml['churn_best_name']} with {ml['churn_accuracy']:.3f} accuracy")
cm = ml['churn_confusion_matrix']
st.write(f"\n Confusion Matrix:")
st.write(f"   True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
st.write(f"   False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

st.write("\n" + "-" * 40)
st.write(" MODEL 2: Customer Satisfaction Level Prediction")
st.write("-" * 40)
st.write(f"\n Multi-class Accuracy: {ml['sat_accuracy']:.3f}")
st.write(f"\n Classification Report:")
st.json(ml['sat_class_report'])

st.write("\n" + "-" * 40)
st.write(" MODEL 3: Sentiment Score Prediction")
st.write("-" * 40)
st.write(f"\n Regression Metrics:")
st.write(f"   Mean Squared Error: {ml['sentiment_mse']:.3f}")
st.write(f"   Mean Absolute Error: {ml['sentiment_mae']:.3f}")
st.write(f"   R² Score: {ml['sentiment_r2']:.3f}")
st.write(f"\n Top Features Influencing Sentiment:")
st.dataframe(ml['sentiment_feature_importance'].head())

# ML Visualizations (original 6-panel figure)
st.write("\n Creating ML Visualizations...")
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Model Comparison Bar Chart
ax1 = axes[0, 0]
models_list = list(ml['churn_cv_results'].keys())
accuracies = [ml['churn_cv_results'][m]['accuracy'] for m in models_list]
bars1 = ax1.bar(models_list, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
for bar, acc in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')

# 2. Feature Importance for Churn
ax2 = axes[0, 1]
top_features = ml['churn_feature_importance'].head(8)
bars2 = ax2.barh(top_features['feature'], top_features['importance'], color='#2E86AB')
ax2.set_title('Top Features for Churn Prediction', fontsize=14, fontweight='bold')
ax2.set_xlabel('Importance')
for bar, imp in zip(bars2, top_features['importance']):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{imp:.3f}', va='center', fontweight='bold')

# 3. Confusion Matrix Heatmap
ax3 = axes[0, 2]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
ax3.set_title('Confusion Matrix - Churn Prediction', fontsize=14, fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Satisfaction Level Distribution
ax4 = axes[1, 0]
sat_counts = df['satisfaction_level'].value_counts()
st.write("current columns in df:",df.columns.tolist())
colors_sat = ['#2E86AB' if x=='High' else '#F18F01' if x=='Medium' else '#A23B72' for x in sat_counts.index]
bars4 = ax4.bar(sat_counts.index, sat_counts.values, color=colors_sat)
ax4.set_title('Satisfaction Level Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Satisfaction Level')
ax4.set_ylabel('Count')
for bar, val in zip(bars4, sat_counts.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontweight='bold')

# 5. Predicted vs Actual Sentiment Scores (we need to recompute predictions for this plot)
X_scaled_full = ml['scaler'].transform(df[ml['feature_columns']])
y_actual = df['composite_score']
y_pred_full = ml['sentiment_regressor'].predict(X_scaled_full)
ax5 = axes[1, 1]
ax5.scatter(y_actual, y_pred_full, alpha=0.6, color='#2E86AB')
ax5.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction')
ax5.set_title('Sentiment Score: Predicted vs Actual', fontsize=14, fontweight='bold')
ax5.set_xlabel('Actual Score')
ax5.set_ylabel('Predicted Score')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. ML Performance Summary
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""
ML MODEL PERFORMANCE SUMMARY

 Churn Prediction (Best: {ml['churn_best_name']})
   Accuracy: {ml['churn_accuracy']:.3f}

 Satisfaction Level Prediction
   Accuracy: {ml['sat_accuracy']:.3f}

 Sentiment Score Prediction
   R² Score: {ml['sentiment_r2']:.3f}
   MAE: {ml['sentiment_mae']:.3f}

 Top Churn Indicators:
1. {ml['churn_feature_importance'].iloc[0]['feature']}: {ml['churn_feature_importance'].iloc[0]['importance']:.3f}
2. {ml['churn_feature_importance'].iloc[1]['feature']}: {ml['churn_feature_importance'].iloc[1]['importance']:.3f}
3. {ml['churn_feature_importance'].iloc[2]['feature']}: {ml['churn_feature_importance'].iloc[2]['importance']:.3f}
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle(' MACHINE LEARNING PERFORMANCE DASHBOARD', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig2)

# --------------------------------------------------------------------------------
# Business Impact Analysis (original)
# --------------------------------------------------------------------------------
st.write("\n" + "=" * 60)
st.write(" BUSINESS IMPACT ANALYSIS")
st.write("=" * 60)

churn_rate = df['churn_risk'].mean()
total_customers = len(df)
avg_order_value = df['order_value'].mean()
clv = avg_order_value * 3
random_savings = total_customers * churn_rate * clv * 0.2
ml_savings = total_customers * churn_rate * clv * 0.7 * ml['churn_accuracy']

st.write(f"\n CUSTOMER METRICS:")
st.write(f"   Total Customers Analyzed: {total_customers}")
st.write(f"   Current Churn Rate: {churn_rate:.1%}")
st.write(f"   Average Customer Value: ${avg_order_value:.2f}")
st.write(f"   Estimated Lifetime Value: ${clv:.2f}")

st.write(f"\n POTENTIAL SAVINGS:")
st.write(f"   Without ML (Random Intervention): ${random_savings:,.2f}")
st.write(f"   With ML (Targeted Intervention): ${ml_savings:,.2f}")
st.write(f"   ADDITIONAL SAVINGS WITH ML: ${ml_savings - random_savings:,.2f}")
st.write(f"   IMPROVEMENT: {((ml_savings - random_savings)/random_savings*100):.0f}%")

high_risk = df[df['churn_risk'] == 1].nlargest(10, 'composite_score')[['customer_name', 'rating', 'sentiment', 'category']]
st.write(f"\n TOP 10 HIGH-RISK CUSTOMERS TO TARGET:")
st.dataframe(high_risk)

# Save ML models (original)
import joblib
joblib.dump(ml['churn_best_model'], 'churn_model.pkl')
joblib.dump(ml['scaler'], 'scaler.pkl')
joblib.dump(ml['label_encoders']['segment'], 'label_encoder_segment.pkl')
joblib.dump(ml['label_encoders']['category'], 'label_encoder_category.pkl')
joblib.dump(ml['label_encoders']['channel'], 'label_encoder_channel.pkl')
joblib.dump(ml['label_encoders']['location'], 'label_encoder_location.pkl')
st.write(f"\n ML Models saved successfully!")
st.write(f"   - churn_model.pkl")
st.write(f"   - scaler.pkl")
st.write(f"   - label_encoders.pkl")

# --------------------------------------------------------------------------------
# Final Summary (original)
# --------------------------------------------------------------------------------
st.write("\n" + "=" * 70)
st.write(" FINAL SUMMARY - SMART FEEDBACK ANALYZER WITH ML")
st.write("=" * 70)
st.write(f"""
 DATASET SUMMARY:
   • Total Feedback Records: {len(df)}
   • Features Used: {len(ml['feature_columns'])}
   • ML Models Trained: 3 (Churn, Satisfaction, Sentiment)

 MODEL PERFORMANCE:
   • Best Churn Prediction: {ml['churn_best_name']} ({ml['churn_accuracy']:.2%} accuracy)
   • Satisfaction Prediction: {ml['sat_accuracy']:.2%} accuracy
   • Sentiment Score R²: {ml['sentiment_r2']:.3f}

 BUSINESS IMPACT:
   • Potential ML Savings: ${ml_savings:,.2f}
   • Improvement over Random: {((ml_savings - random_savings)/random_savings*100):.0f}%
   • High-Risk Customers Identified: {len(high_risk)}

 FILES CREATED:
   • ML Models: 5 model files saved
   • Enhanced Dataset: feedback_analysis_complete.csv
   • JSON Report: analysis_report.json
""")
st.write("=" * 70)

# Provide download buttons for generated files (optional extra)
with st.expander("📥 Download generated files"):
    with open('feedback_analysis_complete.csv', 'rb') as f:
        st.download_button("Download CSV", f, "feedback_analysis_complete.csv")
    with open('analysis_report.json', 'rb') as f:
        st.download_button("Download JSON Report", f, "analysis_report.json")