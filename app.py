from flask import Flask, url_for, request
import pandas as pd
import joblib


app = Flask(__name__)

df = pd.read_csv("rooms.csv")
model, feature_names = joblib.load("occupancy_model.pkl")


def predict_occupancy(room_series):
    features_df = pd.DataFrame([[room_series[f] for f in feature_names]], columns=feature_names)
    prediction = model.predict(features_df)[0]
    return int(prediction)


def render_layout(content_html: str, title: str = "Smart Campus Occupancy"):
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <link
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet"
        >
        <style>
            body {{
                background: #f5f7fb;
            }}
            .navbar-brand span {{
                font-weight: 600;
            }}
            .room-card:hover {{
                box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.08);
                transform: translateY(-2px);
                transition: all 0.15s ease-in-out;
            }}
            .metric-label {{
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
          <div class="container">
            <a class="navbar-brand" href="{url_for('list_rooms')}">
              Smart Campus <span>Occupancy</span>
            </a>
          </div>
        </nav>
        <main class="container mb-5">
          {content_html}
        </main>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """


@app.route('/', methods=["GET"])
def list_rooms():
    rooms = df.to_dict(orient="records")

    rows_html = ""
    for room in rooms:
        link = url_for("get_room", id_value=room["Index"])
        rows_html += f"""
        <tr>
            <td>{room['Index']}</td>
            <td>{room['Temperature']:.2f} °C</td>
            <td>{room['Humidity']:.2f} %</td>
            <td>{room['Light']:.0f} lux</td>
            <td>{room['CO2']:.0f} ppm</td>
            <td>{room['HumidityRatio']:.4f} kgwater-vapor/kg-air</td>
            <td><a class="btn btn-sm btn-primary" href="{link}">Open</a></td>
        </tr>
        """

    content = f"""
    <div class="row mb-4">
      <div class="col">
        <h1 class="h3 mb-1">Demo: Classroom Occupancy</h1>
        <p class="text-muted mb-0">
          Each row represents a snapshot from a classroom. Click a room to see sensor values
          and let the AI model estimate whether it is occupied or free.
        </p>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <strong>Room sensor snapshots</strong>
      </div>
      <div class="table-responsive">
        <table class="table table-hover mb-0 align-middle">
          <thead class="table-light">
            <tr>
              <th>Room ID</th>
              <th>Temperature</th>
              <th>Humidity</th>
              <th>Light</th>
              <th>CO₂</th>
              <th>HumidityRatio</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
    """

    return render_layout(content, title="Rooms – Smart Campus Occupancy")


@app.route('/<int:id_value>', methods=['GET', 'POST'])
def get_room(id_value):
    row = df[df["Index"] == id_value]

    if row.empty:
        return "Room not found", 404

    room = row.iloc[0]
    occupancy = None
    status_badge = ""
    explanation = ""

    if request.method == "POST":
        occupancy = predict_occupancy(room)
        if occupancy == 0:
            status_badge = '<span class="badge bg-success">Likely free</span>'
            explanation = (
                "The model predicts that this room is probably empty based on the current "
                "sensor readings. The building could lower ventilation here to save energy."
            )
        else:
            status_badge = '<span class="badge bg-danger">Likely occupied</span>'
            explanation = (
                "The model predicts that this room is likely occupied. Ventilation and cooling "
                "should be kept higher to maintain air quality and comfort."
            )

    occupancy_section = ""
    if occupancy is not None:
        occupancy_section = f"""
        <div class="alert alert-info mt-3">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <strong>AI decision:</strong> {status_badge}
            </div>
          </div>
          <p class="mb-0 mt-2 small text-muted">{explanation}</p>
        </div>
        """

    content = f"""
    <div class="row mb-3">
      <div class="col-md-8 col-lg-6">
        <div class="card room-card">
          <div class="card-header d-flex justify-content-between align-items-center">
            <div>
              <h2 class="h5 mb-0">Room {id_value}</h2>
              <small class="text-muted">Snapshot from the Kaggle room occupancy dataset</small>
            </div>
            <a href="{url_for('list_rooms')}" class="btn btn-outline-secondary btn-sm">← All rooms</a>
          </div>
          <div class="card-body">
            <div class="row g-3">
              <div class="col-6">
                <div class="metric-label">Temperature</div>
                <div class="fs-5">{room['Temperature']:.2f} °C</div>
              </div>
              <div class="col-6">
                <div class="metric-label">Humidity</div>
                <div class="fs-5">{room['Humidity']:.2f} %</div>
              </div>
              <div class="col-6">
                <div class="metric-label">Light</div>
                <div class="fs-5">{room['Light']:.0f} lux</div>
              </div>
              <div class="col-6">
                <div class="metric-label">CO₂</div>
                <div class="fs-5">{room['CO2']:.0f} ppm</div>
              </div>
              <div class="col-6">
                <div class="metric-label">Humidity ratio</div>
                <div class="fs-5">{room['HumidityRatio']:.4f}</div>
              </div>
            </div>

            <form method="post" class="mt-4">
              <button type="submit" class="btn btn-primary">
                Run AI occupancy check
              </button>
            </form>

            {occupancy_section}
          </div>
        </div>
      </div>
    </div>
    """

    return render_layout(content, title=f"Room {id_value} – Smart Campus Occupancy")


if __name__ == '__main__':
    app.run()