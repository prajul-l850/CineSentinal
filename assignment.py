from ultralytics import solutions,YOLO
import cv2
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--source",
    default="ppl.mp4",
    help="Video source: camera  or video file path"
)
args = parser.parse_args()
yolo = YOLO("yolo11s.pt")

source = int(args.source) if args.source.isdigit() else args.source

cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Error reading video file"


#region = [(640, 300), (1400, 300), (1400, 800), (640, 800)]
region = [(300, 150), (600, 150), (600, 600), (300, 600)]

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    "region_counting.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

counter = solutions.RegionCounter(
    model="yolo11s.pt",
    region=region,
    classes=[0],         
    tracker="bytetrack.yaml",
    conf=0.4,
    verbose=False,
    show=True
)

while cap.isOpened():
    succes, frame = cap.read()
    if not succes:
        print("Video frame is empty or processing is complete.")
        break
    
    c_time = time.strftime("%H:%M:%S")

    results = counter(frame)
    people= list(results.region_counts.values())[0]
    phone_results = yolo(frame, conf=0.4, verbose=False)[0]

    phone = 0
    if phone_results.boxes is not None:
        cls = phone_results.boxes.cls.cpu().numpy()
        phone = (cls == 67).sum()

        print(c_time," | People:", people, "| Phone Detected:", phone)

cap.release()
cv2.destroyAllWindows()
