from flask import Flask, jsonify, url_for, render_template_string, request
import random
import pandas as pd
import joblib


app = Flask(__name__)

df = pd.read_csv("rooms.csv")
model, feature_names = joblib.load("occupancy_model.pkl")

def return_occupancy(room):
    features_df = pd.DataFrame([[room[f] for f in feature_names]], columns=feature_names)
    prediction = model.predict(features_df)[0]
    return int(prediction)
    
    prediction = model.predict(features)[0]
    return int(prediction)


@app.route('/', methods=["GET"])
def list_rooms():
    rooms = df["Index"].tolist()

    html = "<h1>Rooms</h1><ul>"

    for room in rooms:
        link = url_for("get_room", id_value=room)
        html += f'<li><a href="{link}">Room {room}</a></li>'

    html += "</ul>"

    return html

@app.route('/<int:id_value>', methods=['GET', 'POST'])
def get_room(id_value):
    row = df[df["Index"] == id_value]

    if row.empty:
        return "Room not found", 404

    room = row.iloc[0]  # Series, yksi rivi
    occupancy_message = ""

    if request.method == "POST":
        occupancy = return_occupancy(room)  # <-- tärkeää lähettää room
        if occupancy == 0:
            occupancy_message = "<p><strong>The room is likely free.</strong></p>"
        else:
            occupancy_message = "<p><strong>The room is likely occupied.</strong></p>"

    html = f"""
    <h1>Room {id_value}</h1>
    <ul>
        <li><strong>Temperature:</strong> {room['Temperature']} °C</li>
        <li><strong>Humidity:</strong> {room['Humidity']} %</li>
        <li><strong>Light:</strong> {room['Light']} lux</li>
        <li><strong>CO2:</strong> {room['CO2']} ppm</li>
        <li><strong>Humidity Ratio:</strong> {room['HumidityRatio']}</li>
    </ul>

    <form method="post">
        <input type="submit" value="Check Occupancy"/>
    </form>

    {occupancy_message}
    
    <a href="/">Back to all rooms</a>
    """

    return html

# @app.route('/api/model')
# def call_model():

if __name__ == '__main__':
    app.run()