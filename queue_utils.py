import cv2
import numpy as np

# =====================================================
# QUEUE METRICS (BOTTOM 60% OF FRAME)
# =====================================================

def get_queue_metrics(frame, tracked_boxes):
    """
    frame          : Current video frame
    tracked_boxes  : [(x1, y1, x2, y2, track_id), ...]

    returns:
        queue_length
        queue_density
        queue_roi
    """

    h, w, _ = frame.shape

    # ---------------- QUEUE ROI (BOTTOM 60%) ----------------
    x1r = 0
    y1r = int(0.4 * h)
    x2r = w
    y2r = h

    queue_roi = (x1r, y1r, x2r, y2r)
    roi_area = (x2r - x1r) * (y2r - y1r)

    queue_ids = set()
    occupied_area = 0

    # ---------------- VEHICLE CHECK ----------------
    for (x1, y1, x2, y2, track_id) in tracked_boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Check if vehicle center lies inside ROI
        if x1r <= cx <= x2r and y1r <= cy <= y2r:
            queue_ids.add(track_id)
            occupied_area += (x2 - x1) * (y2 - y1)

    # ---------------- METRICS ----------------
    queue_length = len(queue_ids)
    queue_density = min(occupied_area / roi_area, 1.0)

    return queue_length, queue_density, queue_roi


# =====================================================
# DRAW QUEUE ROI ON FRAME
# =====================================================

def draw_queue_roi(frame, queue_roi):
    x1, y1, x2, y2 = queue_roi

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.putText(
        frame,
        "QUEUE ROI (60%)",
        (x1 + 20, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )
