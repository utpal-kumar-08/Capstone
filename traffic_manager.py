import time

PCU_MAPPING = {
    '2-Wheeler': 0.5,
    'Bicycle': 0.2,
    '3-Wheeler': 1.0,
    'Hatchback': 1.0,
    'Sedan': 1.0,
    'SUV': 1.0,
    'MUV': 1.0,
    'Van': 1.0,
    'LCV': 1.5,
    'Mini-bus': 1.5,
    'Tempo-traveller': 1.5,
    'Bus': 3.0,
    'Truck': 3.0,
    'Ambulance': 1.0,
    'Other': 1.0
}

class LaneState:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.pcu_density = 0.0
        self.light_state = "RED"  # RED, GREEN, YELLOW
        self.ambulance_detected = False
        self.accident_detected = False
        self.infractions = [] # List of dicts with id, plate, speed, type
        # tracking variables
        self.objects = {} 
        self.bboxes = {}
        self.speeds = {}
        self.last_ocr_time = 0.0
        self.ocr_cache = {}
        
    def update_pcu(self, detections):
        self.pcu_density = 0.0
        for det in detections:
            v_class = det.get("class", "Other")
            self.pcu_density += PCU_MAPPING.get(v_class, 1.0)
        return self.pcu_density

    def add_infraction(self, objectID, plate_text, speed, v_class):
        exists = any(inf['id'] == objectID for inf in self.infractions)
        if not exists:
            infraction = {
                "id": objectID,
                "lane": self.lane_id + 1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "plate": plate_text,
                "speed": f"{speed:.1f}",
                "type": v_class
            }
            self.infractions.insert(0, infraction)
            self.infractions = self.infractions[:20]
            self._save_to_log(infraction)

    def _save_to_log(self, infraction):
        import json
        import os
        log_file = "infractions_log.json"
        data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
            except:
                data = []
        
        data.insert(0, infraction)
        with open(log_file, "w") as f:
            json.dump(data[:500], f, indent=4) # Keep last 500 entries

class TrafficSignalController:
    def __init__(self, num_lanes=4):
        self.lanes = [LaneState(i) for i in range(num_lanes)]
        self.active_lane_idx = 0
        
        self.police_override_enabled = False
        self.police_detected = False
        
        # State machine settings
        self.state = "GREEN_PHASE" 
        self.last_state_change = time.time()
        
        # Constants
        self.yellow_duration = 3.0
        self.all_red_clearance = 2.0
        self.base_green_time = 5.0 
        self.time_per_pcu = 5.0 
        
        self.current_phase_duration = self._calculate_green_time(self.lanes[0])
        
        for i, lane in enumerate(self.lanes):
            lane.light_state = "GREEN" if i == self.active_lane_idx else "RED"

    def _calculate_green_time(self, lane):
        # Scale green time aggressively based on PCU density
        calculated = self.base_green_time + (lane.pcu_density * self.time_per_pcu)
        # Apply strict bounds: min 5s, max 60s (to prevent starvation)
        return min(max(calculated, 5.0), 60.0)

    def set_police_override_setting(self, enabled: bool):
        self.police_override_enabled = enabled

    def update_signals(self):
        # 1. Police Override Logic
        if self.police_override_enabled and self.police_detected:
            for lane in self.lanes:
                lane.light_state = "YELLOW" # Blink Yellow (simulated) for Manual Traffic Police Control
            return

        # Replace Flash Yellow if exiting Police mode
        if self.lanes[0].light_state == "YELLOW" and self.state == "GREEN_PHASE" and not self.police_detected:
            for i, lane in enumerate(self.lanes):
                lane.light_state = "GREEN" if i == self.active_lane_idx else "RED"
            self.last_state_change = time.time()

        # 2. Ambulance Preemption Logic
        ambulance_lanes = [i for i, lane in enumerate(self.lanes) if lane.ambulance_detected]
        if ambulance_lanes:
            target_lane = ambulance_lanes[0]
            if self.active_lane_idx != target_lane:
                # Force safely transition to ambulance lane
                if self.state == "GREEN_PHASE":
                    self.state = "YELLOW_PHASE"
                    self.last_state_change = time.time()
                    self.current_phase_duration = self.yellow_duration
                    self.lanes[self.active_lane_idx].light_state = "YELLOW"
                elif self.state == "YELLOW_PHASE" and (time.time() - self.last_state_change >= self.current_phase_duration):
                    self.state = "ALL_RED_PHASE"
                    self.last_state_change = time.time()
                    self.current_phase_duration = self.all_red_clearance
                    self.lanes[self.active_lane_idx].light_state = "RED"
                elif self.state == "ALL_RED_PHASE" and (time.time() - self.last_state_change >= self.current_phase_duration):
                    # Transition to Ambulance Lane!
                    self.active_lane_idx = target_lane
                    self.state = "GREEN_PHASE"
                    self.lanes[self.active_lane_idx].light_state = "GREEN"
                    self.last_state_change = time.time()
                    self.current_phase_duration = 60.0 # Ambulance max duration priority
            else:
                # Target lane is already active; stall green light
                self.state = "GREEN_PHASE"
                self.lanes[self.active_lane_idx].light_state = "GREEN"
                self.last_state_change = time.time()
                self.current_phase_duration = 60.0
            return

        # 3. Normal State Machine Cycle
        time_elapsed = time.time() - self.last_state_change
        
        if self.state == "GREEN_PHASE":
            if time_elapsed >= self.current_phase_duration:
                self.state = "YELLOW_PHASE"
                self.last_state_change = time.time()
                self.current_phase_duration = self.yellow_duration
                self.lanes[self.active_lane_idx].light_state = "YELLOW"
                
        elif self.state == "YELLOW_PHASE":
            if time_elapsed >= self.current_phase_duration:
                self.state = "ALL_RED_PHASE"
                self.last_state_change = time.time()
                self.current_phase_duration = self.all_red_clearance
                self.lanes[self.active_lane_idx].light_state = "RED"
                
        elif self.state == "ALL_RED_PHASE":
            if time_elapsed >= self.current_phase_duration:
                # Cycle to next lane
                self.active_lane_idx = (self.active_lane_idx + 1) % len(self.lanes)
                self.state = "GREEN_PHASE"
                self.lanes[self.active_lane_idx].light_state = "GREEN"
                self.last_state_change = time.time()
                self.current_phase_duration = self._calculate_green_time(self.lanes[self.active_lane_idx])
