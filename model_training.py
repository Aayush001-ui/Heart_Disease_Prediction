import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("heart.csv")

# Split into features and target
X = df.drop(columns=['target'])
Y = df['target']

# Train-Test Split (Stratified)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Feature Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression (Balanced Class Weights)
log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_model.fit(X_train_scaled, Y_train)

# Train Optimized Random Forest
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=2)
rf_model.fit(X_train, Y_train)

# Model Evaluation
log_acc = accuracy_score(Y_test, log_model.predict(X_test_scaled))
rf_acc = accuracy_score(Y_test, rf_model.predict(X_test))

log_auc = roc_auc_score(Y_test, log_model.predict_proba(X_test_scaled)[:, 1])
rf_auc = roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1])

print(f"Logistic Regression - Accuracy: {log_acc:.2f}, AUC: {log_auc:.2f}")
print(f"Random Forest - Accuracy: {rf_acc:.2f}, AUC: {rf_auc:.2f}")

# Get feature importances
feature_importance = rf_model.feature_importances_

# Convert to DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

# Save Best Model
best_model = rf_model if rf_auc > log_auc else log_model
joblib.dump(best_model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# SHAP EXPLAINABILITY (For Random Forest)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)  # Get SHAP values

# Debugging Shapes
print("SHAP Values Shape:", np.array(shap_values).shape)  # Should be (N, F, 2)
print("X_test Shape:", X_test.shape)  # Should be (N, F)

# Convert X_test to DataFrame if needed
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Select SHAP values for Class 1 (Heart Disease)
shap_values_class_1 = np.array(shap_values)[:, :, 1]

# Summary Plot (Global Feature Importance)
shap.summary_plot(shap_values_class_1, X_test_df)


# Check distribution of thal values in dataset
a = df.groupby('thal')['target'].value_counts()
print(a)



y_pred = best_model.predict(X_test)  # Standard class predictions (0 or 1)

# If using probability-based models, adjust threshold if needed:
y_pred_prob = best_model.predict_proba(X_test)[:,1]  # Get probabilities for class 1
threshold = 0.5  # Standard threshold
y_pred_adjusted = (y_pred_prob >= threshold).astype(int)  # Convert to labels

# ✅ Step 2: Compute the Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)  # Standard confusion matrix
cm_adjusted = confusion_matrix(Y_test, y_pred_adjusted)

# ✅ Step 3: Visualize Confusion Matrix
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.show()

# Plot standard confusion matrix
plot_confusion_matrix(cm, "Confusion Matrix (Threshold = 0.5)")

print("\n📋 Classification Report:")
print(classification_report(Y_test, y_pred, target_names=["No Disease", "Disease"]))

# Optional: print individual metrics
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"✅ Recall: {recall * 100:.2f}%")
print(f"✅ F1 Score: {f1 * 100:.2f}%")

#Adding correlation Heatmap
plt.figure(figsize=(14,10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
# --- 📈 Done adding Heatmap ---

# Split into features and target
X = df.drop(columns=['target'])
Y = df['target']








# Plot Risk of Heart Attack by Age
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='target', multiple='stack', kde=False)
plt.title('Figure 2: Risk of Heart Attack based on Age')
plt.xlabel('Age')
plt.ylabel('Total Count')
plt.legend(title='Heart Disease', labels=['No Risk of Heart Attack', 'Risk of Heart Attack'])
plt.show()

# Plot Risk by Resting Blood Pressure
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='trestbps', hue='target', multiple='stack', kde=False)
plt.title('Figure 3: Risk of Heart Attack based on Resting Blood Pressure')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Total Count')
plt.legend(title='Heart Disease', labels=['No Risk of Heart Attack', 'Risk of Heart Attack'])
plt.show()










# Pie Chart for Gender vs Heart Disease
male_heart = df[(df['sex'] == 1) & (df['target'] == 1)].shape[0]
female_heart = df[(df['sex'] == 0) & (df['target'] == 1)].shape[0]
male_noheart = df[(df['sex'] == 1) & (df['target'] == 0)].shape[0]
female_noheart = df[(df['sex'] == 0) & (df['target'] == 0)].shape[0]

# Heart Disease Pie
plt.figure(figsize=(6,6))
plt.pie([female_heart, male_heart], labels=['Female Popularity', 'Male Popularity'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('Figure 4: Heart Disease - Gender Distribution')
plt.show()

# No Heart Disease Pie
plt.figure(figsize=(6,6))
plt.pie([female_noheart, male_noheart], labels=['Female Popularity', 'Male Popularity'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('No Heart Disease - Gender Distribution')
plt.show()










# Count by chest pain types
labels = ['Asymptomatic', 'Non-Anginal Pain', 'Atypical Angina', 'Typical Angina']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

heart_cp = df[df['target'] == 1]['cp'].value_counts().sort_index()
noheart_cp = df[df['target'] == 0]['cp'].value_counts().sort_index()

# Heart Disease
plt.figure(figsize=(6,6))
plt.pie(heart_cp, labels=labels, autopct='%1.1f%%', colors=colors)
plt.title('Figure 5: Heart Disease - Chest Pain Type')
plt.show()

# No Heart Disease
plt.figure(figsize=(6,6))
plt.pie(noheart_cp, labels=labels, autopct='%1.1f%%', colors=colors)
plt.title('No Heart Disease - Chest Pain Type')
plt.show()









# Count patients with and without heart disease
heart_disease = df[df['target'] == 1].shape[0]
no_heart_disease = df[df['target'] == 0].shape[0]

# Prepare data
sizes = [no_heart_disease, heart_disease]
labels = ['No Heart Disease', 'Heart Disease']
colors = ['#ff9999', '#66b3ff']

# Plot donut chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

# Add white circle to create donut shape
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Figure 6: Patients Having or Not Having Heart Disease')
plt.tight_layout()
plt.show()