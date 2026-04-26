import numpy as np
import time
from collections import OrderedDict
from scipy.spatial import distance as dist
import supervision as sv

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # next unique object ID
        self.nextObjectID = 0
        # OrderedDicts mapping object ID to its centroid and bounding box
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        # Number of consecutive frames an object has been marked as "disappeared"
        self.disappeared = OrderedDict()

        self.maxDisappeared = max_disappeared
        self.maxDistance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # rects is a list of (startX, startY, endX, endY)
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.bboxes

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputBboxes = {}

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputBboxes[i] = (startX, startY, endX, endY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputBboxes[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # sort details to match
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = inputBboxes[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col], inputBboxes[col])

        return self.objects, self.bboxes

class ByteTrackWrapper:
    """Drop-in replacement for CentroidTracker using supervision's ByteTrack.
    
    Provides the same (objects, bboxes) interface:
      objects = OrderedDict { tracker_id: (cX, cY) }
      bboxes  = OrderedDict { tracker_id: (startX, startY, endX, endY) }
    """
    def __init__(self, track_activation_threshold=0.25, lost_track_buffer=30,
                 minimum_matching_threshold=0.8, frame_rate=20):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, rects):
        """
        Args:
            rects: list of (startX, startY, endX, endY) bounding boxes
        Returns:
            (objects, bboxes) — same format as CentroidTracker.update()
        """
        objects = OrderedDict()
        bboxes = OrderedDict()

        if len(rects) == 0:
            # Feed empty detections so ByteTrack can age out lost tracks
            empty = sv.Detections.empty()
            self.byte_tracker.update_with_detections(empty)
            return objects, bboxes

        xyxy = np.array(rects, dtype=np.float32)
        # All detections get confidence=1.0 since Roboflow already filtered
        confidence = np.ones(len(rects), dtype=np.float32)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
        )

        tracked = self.byte_tracker.update_with_detections(detections)

        if tracked.tracker_id is not None:
            for i, tracker_id in enumerate(tracked.tracker_id):
                startX, startY, endX, endY = tracked.xyxy[i].astype(int)
                cX = int((startX + endX) / 2)
                cY = int((startY + endY) / 2)
                objects[int(tracker_id)] = (cX, cY)
                bboxes[int(tracker_id)] = (int(startX), int(startY), int(endX), int(endY))

        return objects, bboxes

    def reset(self):
        """Reset the tracker state."""
        self.byte_tracker.reset()


class SpeedCalculator:
    def __init__(self, line_a_y, line_b_y, dash_length_m=3.0, gap_length_m=6.0, sequence_pixels=200, lane_id=0):
        self.line_a_y = line_a_y
        self.line_b_y = line_b_y
        self.lane_id = lane_id
        
        # Initial calibration: assume sequence_pixels (default 200) represents 9m cycle
        self.pixels_per_meter = sequence_pixels / (dash_length_m + gap_length_m)
        
        self.object_crossing = {}
        self.object_speeds = {}

    def set_calibration(self, p1, p2, real_distance_m=3.0):
        pixel_gap = abs(p1 - p2)
        if pixel_gap > 0:
            self.pixels_per_meter = pixel_gap / real_distance_m
            print(f"CALIBRATION: Updated PPM to {self.pixels_per_meter:.2f} px/m (gap={pixel_gap}px = {real_distance_m}m)")

    def update(self, objects, video_time_sec=None):
        """
        Args:
            objects: dict of {objectID: (cX, cY)}
            video_time_sec: timestamp from the video in seconds (cv2.CAP_PROP_POS_MSEC / 1000).
                           If None, falls back to time.time() (less accurate for slow processing).
        """
        active_speeds = {}
        top_line = min(self.line_a_y, self.line_b_y)
        bottom_line = max(self.line_a_y, self.line_b_y)
        
        # The real-world distance between the two calibrated lines
        trap_dist_pixels = abs(self.line_a_y - self.line_b_y)
        if self.pixels_per_meter > 0:
            actual_distance_m = trap_dist_pixels / self.pixels_per_meter
        else:
            actual_distance_m = 3.0

        # Use video time if available, otherwise fall back to wall-clock
        current_ts = video_time_sec if video_time_sec is not None else time.time()

        for objectID, centroid in objects.items():
            cX, cY = centroid
            
            if objectID not in self.object_crossing:
                self.object_crossing[objectID] = {
                    "crossed_a": False, "time_a": 0,
                    "crossed_b": False, "time_b": 0,
                    "prev_y": cY, "prev_ts": current_ts
                }
                continue  # Need at least 2 frames to detect a crossing
            
            state = self.object_crossing[objectID]
            prev_y = state["prev_y"]
            prev_ts = state.get("prev_ts", current_ts)
            
            line_a_crossed_this_frame = False
            
            # Detect LINE A crossing: centroid transitions across top_line
            if not state["crossed_a"]:
                if (prev_y < top_line and cY >= top_line) or (prev_y > top_line and cY <= top_line):
                    state["crossed_a"] = True
                    # Interpolate the exact crossing time based on position
                    if cY != prev_y and current_ts != prev_ts:
                        frac = abs(top_line - prev_y) / abs(cY - prev_y)
                        state["time_a"] = prev_ts + frac * (current_ts - prev_ts)
                    else:
                        state["time_a"] = current_ts
                    line_a_crossed_this_frame = True
            
            # Detect LINE B crossing — but NOT in the same frame as Line A
            # This prevents time_taken=0 when vehicle jumps both lines at once
            if not state["crossed_b"] and not line_a_crossed_this_frame:
                if (prev_y < bottom_line and cY >= bottom_line) or (prev_y > bottom_line and cY <= bottom_line):
                    state["crossed_b"] = True
                    # Interpolate the exact crossing time
                    if cY != prev_y and current_ts != prev_ts:
                        frac = abs(bottom_line - prev_y) / abs(cY - prev_y)
                        state["time_b"] = prev_ts + frac * (current_ts - prev_ts)
                    else:
                        state["time_b"] = current_ts
            
            # If vehicle jumped BOTH lines in one frame, estimate using interpolation
            if line_a_crossed_this_frame and not state["crossed_b"]:
                if (prev_y < bottom_line and cY >= bottom_line) or (prev_y > bottom_line and cY <= bottom_line):
                    state["crossed_b"] = True
                    if cY != prev_y and current_ts != prev_ts:
                        frac = abs(bottom_line - prev_y) / abs(cY - prev_y)
                        state["time_b"] = prev_ts + frac * (current_ts - prev_ts)
                    else:
                        # Last resort: estimate time from pixel travel speed
                        state["time_b"] = state["time_a"] + 0.001  # tiny offset to avoid div/0
            
            # If BOTH lines have been crossed, calculate speed
            if state["crossed_a"] and state["crossed_b"]:
                time_taken = abs(state["time_b"] - state["time_a"])
                if time_taken > 0.001:  # Very small threshold since we use interpolation
                    speed_mps = actual_distance_m / time_taken
                    speed_kmh = speed_mps * 3.6
                    
                    # Sanity check: ignore unrealistic speeds (> 300 km/h)
                    if speed_kmh < 300:
                        self.object_speeds[objectID] = speed_kmh
                        print(f"[L{self.lane_id}] SPEED: Vehicle {objectID} = {speed_kmh:.1f} km/h (dist={actual_distance_m:.1f}m, time={time_taken:.3f}s, video_ts={current_ts:.2f}s)")
                
                # Reset for re-measurement
                state["crossed_a"] = False
                state["crossed_b"] = False
            
            state["prev_y"] = cY
            state["prev_ts"] = current_ts

            if objectID in self.object_speeds:
                active_speeds[objectID] = self.object_speeds[objectID]

        # Cleanup departed vehicles
        all_tracked_ids = set(objects.keys())
        for obj_id in list(self.object_crossing.keys()):
            if obj_id not in all_tracked_ids:
                del self.object_crossing[obj_id]
                if obj_id in self.object_speeds:
                    del self.object_speeds[obj_id]

        return active_speeds

class AccidentVerificationEngine:
    def __init__(self, verification_buffer_sec=2.5, iou_threshold=0.3):
        self.verification_buffer_sec = verification_buffer_sec
        self.iou_threshold = iou_threshold
        # Track when pairs of vehicles first started overlapping: {(id1, id2): timestamp}
        self.overlapping_pairs = {}
        
    def check_accident(self, objects, bboxes, current_speeds):
        """
        objects: {id: (cX, cY)}
        bboxes: {id: (startX, startY, endX, endY)}
        current_speeds: {id: speed_kmh}
        
        Returns: boolean indicating if a confirmed accident is happening
        """
        current_time = time.time()
        object_ids = list(bboxes.keys())
        current_overlaps = set()
        
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                id1 = object_ids[i]
                id2 = object_ids[j]
                
                # Consistent ordering for dictionary keys
                pair = (min(id1, id2), max(id1, id2))
                
                bb1 = bboxes[id1]
                bb2 = bboxes[id2]
                
                # Calculate Intersection over Union (IoU)
                x_left = max(bb1[0], bb2[0])
                y_top = max(bb1[1], bb2[1])
                x_right = min(bb1[2], bb2[2])
                y_bottom = min(bb1[3], bb2[3])
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
                    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
                    iou = intersection / float(area1 + area2 - intersection)
                    
                    if iou > self.iou_threshold:
                        current_overlaps.add(pair)
                        if pair not in self.overlapping_pairs:
                            self.overlapping_pairs[pair] = current_time
                        else:
                            # If overlapping persists longer than the buffer, trigger accident
                            if current_time - self.overlapping_pairs[pair] >= self.verification_buffer_sec:
                                print(f"[WARNING] Mathematical Accident Triggered! Vehicles {id1} and {id2} overlapping for > {self.verification_buffer_sec}s")
                                return True
                                
        # Cleanup pairs that are no longer overlapping
        for pair in list(self.overlapping_pairs.keys()):
            if pair not in current_overlaps:
                del self.overlapping_pairs[pair]
                
        return False
