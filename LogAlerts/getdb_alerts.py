from flask import Flask, jsonify, request, render_template
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client['SecurityAlerts']
collection = db['Detections']

# ---------------------------
# Get alerts with optional filters
# ---------------------------
@app.route('/get_alerts_filtered', methods=['GET'])
def get_alerts_filtered():
    date = request.args.get('date')   # e.g. 2025-09-25
    time = request.args.get('time')   # e.g. 07:17:24

    query = {}
    if date:
        query["date"] = date
    if time:
        query["time"] = time

    alerts = list(collection.find(query).sort("_id", -1))
    for alert in alerts:
        alert["_id"] = str(alert["_id"])
    return jsonify(alerts)


# ---------------------------
# Frontend Page
# ---------------------------
@app.route('/')
def index():
    return render_template("alerts.html")


if __name__ == "__main__":
    app.run(debug=True, port=5050)  # ✅ Running on port 5050
