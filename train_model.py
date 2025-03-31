from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Ensure the 'static' directory exists for storing graphs
if not os.path.exists("static"):
    os.makedirs("static")

# Load the XGBoost models
with open("model/xgboost_model.pkl", "rb") as f:
    models = pickle.load(f)

def predict_traffic(date, time, junction, is_holiday):
    datetime_str = f"{date} {time}"
    datetime_obj = pd.to_datetime(datetime_str)

    hour = datetime_obj.hour

    # Creating the required feature set for prediction
    traffic_lag_1 = 0  # Placeholder
    traffic_lag_24 = 0  # Placeholder
    rolling_mean_3 = 0  # Placeholder
    rolling_std_3 = 0  # Placeholder

    input_data = pd.DataFrame([[hour, traffic_lag_1, traffic_lag_24, rolling_mean_3, rolling_std_3]],
                              columns=["Hour", "traffic_lag_1", "traffic_lag_24", "rolling_mean_3", "rolling_std_3"])

    # Access model using index
    model = models[junction - 1]
    predicted_vehicles = model.predict(input_data)[0]
    traffic_status = "High" if predicted_vehicles > 50 else "Low"
    recommendation = "Avoid this junction" if traffic_status == "High" else "You can use this junction"

    # Generate sample actual data (replace with real data if available)
    actual_vehicles = [predicted_vehicles * 0.9, predicted_vehicles * 1.1]  # Example actual values

    # Plot the graph
    plt.figure(figsize=(6, 4))
    plt.plot(["Actual", "Predicted"], [actual_vehicles[0], predicted_vehicles], marker='o', linestyle='--', color='b', label="Actual vs Predicted")
    plt.xlabel("Category")
    plt.ylabel("Number of Vehicles")
    plt.title(f"Traffic Flow Prediction for Junction {junction}")
    plt.legend()
    plt.grid(True)

    # Save the graph
    graph_filename = f"static/traffic_graph_{junction}.png"
    plt.savefig(graph_filename)
    plt.close()

    return predicted_vehicles, traffic_status, recommendation, graph_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    date = request.form['date']
    time = request.form['time']
    junction = int(request.form['junction'])
    is_holiday = 1 if request.form.get('is_holiday') == 'on' else 0

    predicted_vehicles, traffic_status, recommendation, graph_filename = predict_traffic(date, time, junction, is_holiday)

    return render_template('result.html', 
                           predicted_vehicles=predicted_vehicles, 
                           traffic_status=traffic_status, 
                           recommendation=recommendation, 
                           graph_filename=graph_filename)

if __name__ == '__main__':
    app.run(debug=True)