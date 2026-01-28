import time
import math

# ---------------- RED LIGHT JUMP ----------------
def detect_red_light_jump(vehicle_box, stop_line_y, light_is_red):
    """
    vehicle_box: (x1, y1, x2, y2, id)
    """
    x1, y1, x2, y2, vid = vehicle_box
    vehicle_bottom = y2

    if light_is_red and vehicle_bottom > stop_line_y:
        return True
    return False


# ---------------- RASH DRIVING ----------------
speed_memory = {}

def detect_rash_driving(vehicle_box, fps, speed_threshold=35):
    """
    Simple pixel-based speed estimation
    """
    x1, y1, x2, y2, vid = vehicle_box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    if vid not in speed_memory:
        speed_memory[vid] = (cx, cy, time.time())
        return False

    px, py, pt = speed_memory[vid]
    dist = math.hypot(cx - px, cy - py)
    dt = time.time() - pt

    speed = (dist / dt) if dt > 0 else 0
    speed_memory[vid] = (cx, cy, time.time())

    return speed > speed_threshold
