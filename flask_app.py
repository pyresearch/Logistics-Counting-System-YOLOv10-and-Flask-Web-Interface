import cv2
import os
import supervision as sv
from ultralytics import YOLOv10
from flask import Flask, render_template, Response, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv10 model (ensure this is trained with your custom dataset)
model = YOLOv10("last.pt")

# Replace category_dict with your custom class names
category_dict = {
    0: 'barcode', 1: 'car', 2: 'cardboard box', 3: 'fire', 4: 'forklift',
    5: 'freight container', 6: 'gloves', 7: 'helmet', 8: 'ladder', 9: 'license plate',
    10: 'person', 11: 'qr code', 12: 'road sign', 13: 'safety vest', 14: 'smoke',
    15: 'traffic cone', 16: 'traffic light', 17: 'truck', 18: 'van', 19: 'wood pallet'
}

# Dictionary to hold the count of detected objects
object_counts = {category: 0 for category in category_dict.values()}

# Global variable to store video source (0 for webcam, or path for video file)
video_source = "video3.mp4"

def generate_frames():
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video source.")
    
    # Define a VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    bounding_box_annotator = sv.BoundingBoxAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the YOLOv10 model on the current frame
        results = model(frame)[0]

        # Convert results to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Reset object count for each frame
        global object_counts
        object_counts = {category: 0 for category in category_dict.values()}

        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict.get(class_id, "Unknown")
            object_counts[class_name] += 1
            
            # Annotate the detection on the frame
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)  # Increased text size

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 300), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        y_offset = 40
        for class_name, count in object_counts.items():
            if count > 0:
                cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)  # Increased text size
                y_offset += 40  # Adjusted spacing
        
        # Write the processed frame to the video file
        out.write(frame)

        # Encode the frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for Flask streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()  # Release the VideoWriter

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_counts')
def get_object_counts():
    return jsonify(object_counts)

@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global video_source
    data = request.json
    if 'video_source' in data:
        if data['video_source'] == 'webcam':
            video_source = 0
        else:
            video_source = data['video_source']
    return jsonify({'status': 'success', 'video_source': video_source})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
