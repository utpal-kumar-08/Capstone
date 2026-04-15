import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from traffic_manager import TrafficSignalController

def test_logging():
    controller = TrafficSignalController(num_lanes=4)
    lane = controller.lanes[0]
    
    print("Adding test infraction...")
    lane.add_infraction("test_obj_1", "ABC-123", 75.5, "Car")
    
    log_file = "infractions_log.json"
    if os.path.exists(log_file):
        print(f"Success! {log_file} created.")
        with open(log_file, "r") as f:
            print("Content:")
            print(f.read())
    else:
        print("Failure: Log file not created.")

if __name__ == "__main__":
    test_logging()
