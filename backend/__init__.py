from flask import Flask, request, jsonify, render_template
import numpy as np

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from flask_cors import CORS

app = Flask(
    __name__,
    template_folder="C:/Users/Msys/Desktop/Disease-prediction(ML project)/frontend/templates",
    static_folder="C:/Users/Msys/Desktop/Disease-prediction(ML project)/frontend/static",
)
CORS(app)


# Reading the train.csv by removing the
# last column since it's an empty column
training_data = pd.read_csv("backend/data/symptoms_Data_Training.csv")
training_data.drop(training_data.columns[-1], axis=1, inplace=True)


def data_processing(data):
    # List of target labels
    target_labels = [
        "Diabetes",
        "Hypertension",
        "Bronchial Asthma",
        "Allergy",
        "Common Cold",
    ]

    # 1. Change all rows not in target labels to "not ill"
    data["prognosis"] = data["prognosis"].apply(
        lambda x: x if x in target_labels else "not ill"
    )

    return data

data = data_processing(training_data)
# Encoding the target value into numerical value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

XX = data.iloc[:, :-1]
y = data.iloc[:, -1]

sm = SMOTE(random_state=4)

XX, y = sm.fit_resample(XX, y)

# Assume X is your feature matrix and y is your target variable
selector = SelectKBest(score_func=chi2, k=16)  # Select top 10 features
X = selector.fit_transform(XX, y)

selected_columns = XX.columns[selector.get_support()].tolist()
  
@app.route("/")
def home():

    return render_template("home.html", symptoms=selected_columns)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # test data preprocessing
        test_data = pd.read_csv("backend/data/symptoms_Data_Testing.csv")
        test_data = data_processing(test_data)
        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])

        test_X, test_Y = sm.fit_resample(test_X, test_Y)

        test_X = selector.fit_transform(test_X, test_Y)
        selected_columns = XX.columns[selector.get_support()].tolist()

        # Training the models on whole data
        final_svm_model = SVC(C=0.1, gamma=1)
        final_rf_model = RandomForestClassifier(
            random_state=18, max_depth=3, max_leaf_nodes=6
        )
        final_dt_model = DecisionTreeClassifier(
            criterion="entropy", max_depth=6, random_state=1
        )

        final_svm_model.fit(X, y)
        final_rf_model.fit(X, y)
        final_dt_model.fit(X, y)

        # Making prediction by take mode of predictions
        # made by all the classifiers
        svm_preds = final_svm_model.predict(test_X)
        rf_preds = final_rf_model.predict(test_X)
        dt_preds = final_dt_model.predict(test_X)

        from scipy import stats

        final_preds = [
            stats.mode([i, j, k])[0] for i, j, k in zip(svm_preds, dt_preds, rf_preds)
        ]

        print(
            f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds)*100}\
            precision_score on Test dataset by the combined model: {precision_score(test_Y, final_preds,average="weighted")*100}\
            recall_score on Test dataset by the combined model: {recall_score(test_Y, final_preds,average="weighted")*100}\
            f1_score on Test dataset by the combined model: {f1_score(test_Y, final_preds,average="weighted")*100}"
        )

        symptoms = selected_columns

        user_symptoms = request.json.get("symptoms")

        # Creating a symptom index dictionary to encode the
        # input symptoms into numerical form
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom_index[value] = index

        data_dict = {
            "symptom_index": symptom_index,
            "predictions_classes": encoder.classes_,
        }

        # Defining the Function
        # Input: string containing symptoms separated by commas
        # Output: Generated predictions by models
        def predictDisease(symptoms):
            # creating input data for the models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1

            # reshaping the input data and converting it
            # into suitable format for model predictions
            input_data = np.array(input_data).reshape(1, -1)

            # generating individual outputs
            rf_prediction = data_dict["predictions_classes"][
                final_rf_model.predict(input_data)[0]
            ]
            svm_prediction = data_dict["predictions_classes"][
                final_svm_model.predict(input_data)[0]
            ]
            dt_prediction = data_dict["predictions_classes"][
                final_dt_model.predict(input_data)[0]
            ]

            # making final prediction by taking mode of all predictions
            # Use statistics.mode instead of scipy.stats.mode
            import statistics

            final_prediction = statistics.mode(
                [rf_prediction, svm_prediction, dt_prediction]
            )

            predictions = {
                "rf_model_prediction": rf_prediction,
                "svm_model_prediction": svm_prediction,
                "dt_model_prediction": dt_prediction,
                "final_prediction": final_prediction,
            }

            return predictions

        # Testing the function
        predictions = predictDisease(user_symptoms)

        print(predictions)

        return jsonify(
            {
                "rf_model_prediction": predictions["rf_model_prediction"],
                "svm_model_prediction": predictions["svm_model_prediction"],
                "dt_model_prediction": predictions["dt_model_prediction"],
                "final_prediction": predictions["final_prediction"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
