from ultralytics import YOLO
import cv2
import numpy as np

# VIDEO_PATH = "/content/traffic.mp4"
# AUTO_MODEL_PATH = "/content/auto_best.pt"
# COCO_MODEL_PATH = "yolov8s.pt"
# OUTPUT_VIDEO = "/content/final_gpu_balanced.mp4"

VIDEO_PATH = "/kaggle/input/mp4-file/15040621_2160_3840_30fps.mp4"
AUTO_MODEL_PATH = "/kaggle/input/auto-detecting-ml-model/pytorch/default/1/best.pt"
COCO_MODEL_PATH = "yolov8s.pt"

OUTPUT_VIDEO = "/kaggle/working/final_output_gpu_v4.mp4"

CONF_MARGIN = 0.15
IOU_MATCH = 0.5
COCO_VEHICLES = {"car","bus","truck","motorcycle"}

coco_model = YOLO(COCO_MODEL_PATH)
auto_model = YOLO(AUTO_MODEL_PATH)

def iou(a,b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1])
    xB=min(a[2],b[2]); yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-6)

cap=cv2.VideoCapture(VIDEO_PATH)
fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out=cv2.VideoWriter(OUTPUT_VIDEO,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))

print("Running GPU-balanced pipeline...")

while True:
    ret,frame=cap.read()
    if not ret: break

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # ===== COCO DETECTION + TRACKING ON GPU =====
    coco_res=coco_model.track(frame_rgb,persist=True,conf=0.25,verbose=False)[0]

    vehicle_boxes=[]
    if coco_res.boxes is not None:
        for b in coco_res.boxes:
            label=coco_res.names[int(b.cls)]
            conf=float(b.conf)
            if label in COCO_VEHICLES:
                x1,y1,x2,y2=map(int,b.xyxy[0])
                vehicle_boxes.append((x1,y1,x2,y2,label,conf))

    # ===== Batch crops for AUTO model =====
    crops=[]
    crop_coords=[]
    for (x1,y1,x2,y2,_,_) in vehicle_boxes:
        crop=frame_rgb[y1:y2,x1:x2]
        if crop.size>0:
            crops.append(crop)
            crop_coords.append((x1,y1))

    auto_results=[]
    if len(crops)>0:
        auto_results=auto_model.predict(crops,conf=0.25,verbose=False)

    tracked_autos=[]
    for i,res in enumerate(auto_results):
        if res.boxes is None: continue
        xoff,yoff=crop_coords[i]
        for b in res.boxes:
            ax1,ay1,ax2,ay2=b.xyxy[0]
            tracked_autos.append((int(xoff+ax1),int(yoff+ay1),
                                  int(xoff+ax2),int(yoff+ay2),
                                  "auto",float(b.conf)))

    # ===== CONFIDENCE FUSION =====
    final_detections=[[x1,y1,x2,y2,l,c,"coco"] for (x1,y1,x2,y2,l,c) in vehicle_boxes]

    for (ax1,ay1,ax2,ay2,alabel,aconf) in tracked_autos:
        best_iou=0; best_idx=-1
        for i,(cx1,cy1,cx2,cy2,clabel,cconf,src) in enumerate(final_detections):
            overlap=iou((ax1,ay1,ax2,ay2),(cx1,cy1,cx2,cy2))
            if overlap>best_iou: best_iou=overlap; best_idx=i

        if best_iou>=IOU_MATCH:
            coco_conf=final_detections[best_idx][5]
            if aconf>=coco_conf+CONF_MARGIN:
                final_detections[best_idx]=[ax1,ay1,ax2,ay2,"auto",aconf,"auto"]
        else:
            final_detections.append([ax1,ay1,ax2,ay2,"auto",aconf,"auto"])

    # ===== DRAW =====
    for (x1,y1,x2,y2,label,conf,_) in final_detections:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(frame,f"{label.upper()} {conf:.2f}",
                    (x1,max(30,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)

    out.write(frame)

cap.release(); out.release()
print("Done.")