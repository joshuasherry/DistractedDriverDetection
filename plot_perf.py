import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time
import os

LOG_FILE = "perf_log.csv"

if not os.path.exists(LOG_FILE):
    print(f"No log file found: {LOG_FILE}")
    exit()

# ------------------------
# Live plotting setup
# ------------------------
plt.ion()  # interactive mode

# Keep last N points for live view
N = 200
efficiency_history = deque(maxlen=N)
time_history = deque(maxlen=N)

fig, ax = plt.subplots()
line, = ax.plot([], [], label="Efficiency Score")
ax.set_ylim(0, 5000)
ax.set_xlabel("Frame")
ax.set_ylabel("Efficiency Score")
ax.set_title("Live Efficiency Score")
ax.grid(True)
ax.legend()

# ------------------------
# Load log in streaming manner
# ------------------------
last_row_count = 0

while True:
    try:
        df = pd.read_csv(LOG_FILE)
        # only take new rows
        new_rows = df.iloc[last_row_count:]
        last_row_count = len(df)

        for _, row in new_rows.iterrows():
            fps = row["fps"]
            landmark_ms = row["landmark_ms"]
            cpu_percent = row.get("cpu", 50)  # if CPU not logged, assume 50%

            # Avoid divide by zero
            if landmark_ms <= 0 or cpu_percent <= 0:
                efficiency = 0
            else:
                efficiency = fps / (landmark_ms * cpu_percent) * 1000

            efficiency_history.append(efficiency)
            time_history.append(len(efficiency_history))

        # Update live plot
        line.set_xdata(range(len(efficiency_history)))
        line.set_ydata(efficiency_history)
        ax.set_xlim(0, len(efficiency_history))
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)  # adjust refresh speed

    except KeyboardInterrupt:
        print("Exiting live plot...")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.5)

# ------------------------
# Offline plot once finished
# ------------------------
plt.ioff()
plt.figure(figsize=(10,5))
plt.plot(range(len(efficiency_history)), efficiency_history, label="Efficiency Score")
plt.xlabel("Frame")
plt.ylabel("Efficiency Score")
plt.title("Efficiency Score Over Time")
plt.grid(True)
plt.legend()
plt.show()
