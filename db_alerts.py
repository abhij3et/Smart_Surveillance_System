from flask import jsonify
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['SecurityAlerts']
collection = db['Detections']

@app.route('/get_alerts')
def get_alerts():
    # Fetch last 50 alerts sorted by newest first
    alerts = list(collection.find().sort("_id", -1).limit(50))
    for alert in alerts:
        alert['_id'] = str(alert['_id'])  # Convert ObjectId to string
        if 'image' in alert:
            alert['image'] = f"data:image/jpeg;base64,{alert['image'].decode()}"  # convert binary to base64 for frontend
    return jsonify(alerts)
