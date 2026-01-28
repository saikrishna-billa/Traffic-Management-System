import streamlit as st
import cv2
import tempfile
import time
import pandas as pd

from v4_auto import detect_and_track
from queue_utils import get_queue_metrics, draw_queue_roi

from violations import detect_red_light_jump, detect_rash_driving

st.set_page_config(page_title="Traffic Violation System", layout="wide")
st.title("🚦 Smart Traffic Monitoring & Violation Detection")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_col, info_col = st.columns([3, 1])
    frame_box = video_col.empty()

    qlen_m = info_col.metric("Queue Length", 0)
    qden_m = info_col.metric("Queue Density", 0.0)
    viol_m = info_col.metric("Violations", 0)

    csv_rows = []
    frame_no = 0

    STOP_LINE_Y = 300
    light_is_red = True  # (You can connect this to signal logic)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        tracked = detect_and_track(frame)

        q_len, q_den, roi = get_queue_metrics(frame, tracked)
        draw_queue_roi(frame, roi)

        violations = []

        for box in tracked:
            x1, y1, x2, y2, vid = box

            red_jump = detect_red_light_jump(box, STOP_LINE_Y, light_is_red)
            rash = detect_rash_driving(box, fps)

            if red_jump or rash:
                violations.append(vid)

                color = (0, 0, 255)
                label = "RED JUMP" if red_jump else "RASH"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"ID {vid}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            csv_rows.append({
                "frame": frame_no,
                "vehicle_id": vid,
                "queue_length": q_len,
                "queue_density": round(q_den, 2),
                "red_light_jump": red_jump,
                "rash_driving": rash
            })

        qlen_m.metric("Queue Length", q_len)
        qden_m.metric("Queue Density", round(q_den, 2))
        viol_m.metric("Violations", len(set(violations)))

        cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y),
                 (0, 0, 255), 3)

        frame_box.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        time.sleep(1 / fps)

    cap.release()

    df = pd.DataFrame(csv_rows)
    st.success("Processing completed")

    st.download_button(
        "⬇️ Download Violation CSV",
        df.to_csv(index=False).encode("utf-8"),
        "traffic_violations.csv",
        "text/csv"
    )
