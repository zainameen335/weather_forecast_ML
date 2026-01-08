import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Rain Prediction Dashboard 🌦️")


# Section 1: Single Model (Random Forest)

st.header("Random Forest Rain Prediction")


df = pd.read_csv("https://raw.githubusercontent.com/zainameen335/rfm-machine-learning-project/master/weatherAUS.csv")
st.write("Dataset Preview:", df.head())


features = df.iloc[:, [1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
    
results = df.iloc[:, -1].values.reshape(-1,1)

imputer = SimpleImputer(strategy="most_frequent")
features = imputer.fit_transform(features)
results = imputer.fit_transform(results)

le_cols = [0,4,6,7,-1]
le = LabelEncoder()
for col in le_cols:
    features[:, col] = le.fit_transform(features[:, col])
results[:,-1] = le.fit_transform(results[:,-1])

sample_idx = df.sample(frac=0.8, random_state=0).index
features = features[sample_idx]
results = results[sample_idx]

x_train, x_test, y_train, y_test = train_test_split(features, results, test_size=0.2, random_state=0)
y_train = y_train.ravel().astype(int)
y_test = y_test.ravel().astype(int)

with st.spinner("Training model... This may take a few seconds"):
    classifier = RandomForestClassifier(
          n_estimators=150, 
          class_weight='balanced', 
          max_depth=12, 
          min_samples_leaf=5, 
          random_state=0
       )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['No','Yes']))

    feature_names = [
        'Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine',
        'WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am',
        'WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
        'Cloud9am','Cloud3pm','RainToday'
    ]
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8,6))
    plt.barh(imp_df['Feature'][:10], imp_df['Importance'][:10])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Important Features")
    st.pyplot(plt.gcf())

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No Rain','Rain'], yticklabels=['No Rain','Rain'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt.gcf())

    correct = np.sum(y_test == y_pred)
    incorrect = np.sum(y_test != y_pred)
    fig3, ax3 = plt.subplots()
    ax3.bar(['Correct','Incorrect'], [correct, incorrect])
    ax3.set_title("Model Prediction Performance")
    st.pyplot(fig3)

# Section 2: Model Comparison

st.header("Compare Multiple Models")
if st.button("Show Model Comparison"):
    df = pd.read_csv("https://raw.githubusercontent.com/zainameen335/rfm-machine-learning-project/master/weatherAUS.csv")
    features = df.iloc[:, [1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
    results = df.iloc[:, -1].values.reshape(-1,1)

    imputer = SimpleImputer(strategy="most_frequent")
    features = imputer.fit_transform(features)
    results  = imputer.fit_transform(results)

    le_cols = [0,4,6,7,-1]
    le = LabelEncoder()
    for col in le_cols:
        features[:, col] = le.fit_transform(features[:, col])
    results[:,-1] = le.fit_transform(results[:,-1])

    sc = StandardScaler()
    features = sc.fit_transform(features)

    sample_idx = df.sample(frac=0.3, random_state=0).index
    features = features[sample_idx]
    results = results[sample_idx]

    x_train, x_test, y_train, y_test = train_test_split(features, results, test_size=0.2, random_state=0)
    y_train = y_train.ravel().astype(int)
    y_test = y_test.ravel().astype(int)

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            max_depth=12,
            min_samples_leaf=5,
            random_state=0
        ),
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=0
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=(np.sum(y_train==0)/np.sum(y_train==1)),
            random_state=0
        )
    }
    with st.spinner("Training models... This may take a few minutes"):
     results_list = []
     for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=['No','Yes'], output_dict=True)
        results_list.append({
            "Model": name,
            "Accuracy": report['accuracy'],
            "Recall Yes": report['Yes']['recall'],
            "Precision Yes": report['Yes']['precision'],
            "F1 Yes": report['Yes']['f1-score']
        })

    df_results = pd.DataFrame(results_list)
    st.write("Model Comparison Results:", df_results)

    # Plot metrics
    df_plot = df_results.melt(id_vars='Model', value_vars=['Recall Yes', 'Precision Yes', 'F1 Yes'])
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_plot, x='Model', y='value', hue='variable')
    plt.ylim(0,1)
    plt.title("Rain Detection Metrics Comparison")
    st.pyplot(plt.gcf())

