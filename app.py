from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import geopy.distance
import datetime

app = Flask(__name__)

# Load XGBoost models
with open("model/xgboost_model.pkl", "rb") as f:
    models = pickle.load(f)

# Predefined junction locations (latitude, longitude)
junctions = {
    1: (12.971598, 77.594566),
    2: (12.935223, 77.624012),
    3: (12.927923, 77.627108),
    4: (12.914142, 77.646333)
}

def find_nearest_junction(lat, lon):
    user_location = (lat, lon)
    min_distance = float('inf')
    nearest_junction = None
    
    for junction, coords in junctions.items():
        distance = geopy.distance.geodesic(user_location, coords).km
        if distance < min_distance:
            min_distance = distance
            nearest_junction = junction
    
    return nearest_junction

def predict_traffic(date, time, junction, is_holiday):
    datetime_str = f"{date} {time}"
    datetime_obj = pd.to_datetime(datetime_str)
    hour = datetime_obj.hour
    
    traffic_lag_1, traffic_lag_24 = 0, 0
    rolling_mean_3, rolling_std_3 = 0, 0
    
    input_data = pd.DataFrame([[hour, traffic_lag_1, traffic_lag_24, rolling_mean_3, rolling_std_3]],
                              columns=["Hour", "traffic_lag_1", "traffic_lag_24", "rolling_mean_3", "rolling_std_3"])
    
    model = models[junction - 1]
    predicted_vehicles = model.predict(input_data)[0]
    traffic_status = "High" if predicted_vehicles > 50 else "Low"
    recommendation = "Avoid this junction" if traffic_status == "High" else "You can use this junction"
    
    return predicted_vehicles, traffic_status, recommendation, junction

def generate_prediction_plot(junction):
    time_range = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(48, -1, -1)]
    actual_traffic = [30 + i % 10 for i in range(49)]
    predicted_traffic = [35 + i % 8 for i in range(49)]
    
    rolling_mean = pd.Series(actual_traffic).rolling(window=3).mean()
    rolling_std = pd.Series(actual_traffic).rolling(window=3).std()
    
    plt.figure(figsize=(8, 6))
    plt.plot(time_range, actual_traffic, label='Actual', color='blue')
    plt.plot(time_range, predicted_traffic, label='Predicted', color='red')
    plt.plot(time_range, rolling_mean, label='Mean', linestyle='dashed', color='green')
    plt.fill_between(time_range, rolling_mean - rolling_std, rolling_mean + rolling_std, color='gray', alpha=0.3)
    
    plt.xlabel("Time")
    plt.ylabel("Vehicle Count")
    plt.title(f"Traffic Prediction for Junction {junction} (Last 48 Hours)")
    plt.legend()
    
    plot_path = "static/prediction_plot.png"
    plt.xticks(rotation=45)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    date = request.form['date']
    time = request.form['time']
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    is_holiday = 1 if request.form.get('is_holiday') == 'on' else 0
    
    junction = find_nearest_junction(latitude, longitude)
    predicted_vehicles, traffic_status, recommendation, junction = predict_traffic(date, time, junction, is_holiday)
    plot_path = generate_prediction_plot(junction)
    
    return render_template('result.html', predicted_vehicles=predicted_vehicles, 
                           traffic_status=traffic_status, recommendation=recommendation,
                           junction=junction, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
