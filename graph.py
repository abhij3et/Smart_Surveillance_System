import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict
from datetime import datetime
from flask import Flask, Response
import io

app = Flask(__name__)

# --- MongoDB Configuration ---
try:
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["SecurityAlerts"]
    collection = db["Detections"]
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

@app.route('/get_analytics_plot')
def get_analytics_plot():
    try:
        # --- Query and Process Data ---
        all_detections = collection.find({})
        detection_type_counts = {
            "Crowd": 0,
            "Weapon": 0,
            "Fight": 0
        }
        detection_trends_daily = defaultdict(lambda: defaultdict(int))
        
        for detection in all_detections:
            alert_type = detection.get("type")
            detection_date_str = detection.get("date")

            if alert_type in detection_type_counts:
                detection_type_counts[alert_type] += 1
            
            if detection_date_str:
                detection_trends_daily[detection_date_str][alert_type] += 1

        # --- Prepare Data for Plotting ---
        labels = list(detection_type_counts.keys())
        sizes = [detection_type_counts[label] for label in labels]
        colors = ['#1f77b4', '#d62728', '#2ca02c']

        sorted_dates = sorted(detection_trends_daily.keys())
        crowd_counts_over_time = [detection_trends_daily[date]["Crowd"] for date in sorted_dates]
        weapon_counts_over_time = [detection_trends_daily[date]["Weapon"] for date in sorted_dates]
        fight_counts_over_time = [detection_trends_daily[date]["Fight"] for date in sorted_dates]

        # --- Create Visualizations using Matplotlib ---
        plt.style.use('dark_background')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

        # Subplot 1: Pie Chart
        ax1 = axes[0]
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                                           wedgeprops=dict(width=0.4, edgecolor='w'))
        ax1.set_title("Detection Types", color='white', fontsize=16)
        ax1.axis('equal')
        for autotext in autotexts: autotext.set_color('white')
        for text in texts: text.set_color('white')

        # Subplot 2: Bar Chart
        ax2 = axes[1]
        bar_width = 0.2
        index = range(len(sorted_dates))
        ax2.bar([i - bar_width for i in index], crowd_counts_over_time, bar_width, label='Crowd Detections', color=colors[0])
        ax2.bar(index, weapon_counts_over_time, bar_width, label='Weapon Detections', color=colors[1])
        ax2.bar([i + bar_width for i in index], fight_counts_over_time, bar_width, label='Fight Detections', color=colors[2])
        ax2.set_title("Detection Trends Over Time", color='white', fontsize=16)
        ax2.set_xlabel("Date", color='white')
        ax2.set_ylabel("Number of Detections", color='white')
        ax2.set_xticks(index)
        ax2.set_xticklabels(sorted_dates, rotation=45, ha='right', color='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax2.grid(axis='y', linestyle='--', alpha=0.7, color='gray')

        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='black')
        img_buffer.seek(0)
        plt.close(fig)

        return Response(img_buffer.getvalue(), mimetype='image/png')
    except Exception as e:
        print(f"Error generating plot: {e}")
        return Response("Error generating plot", status=500)

if __name__ == '__main__':
    app.run(debug=True, port=5001)