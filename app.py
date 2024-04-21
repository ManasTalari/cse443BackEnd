from flask import Flask, Response, request
import cv2
from flask_cors import CORS 
import pytesseract
import numpy as np

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://cse443Prudhvi:pEjVbWv6oJHaHSJA@cluster0.7ournot.mongodb.net/"
# Create a new client and connect to the server


cluster = MongoClient(uri)
db = cluster['cseData']  # Replace '<dbname>' with your actual database name
collection = db['numberPlates'] 



FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PLATE_CASCADE = cv2.CascadeClassifier('./indian_license_plate.xml')
MIN_AREA = 200
COLOR = (255, 0, 255) 


# Declare a variable with the type ndarray[Any, dtype[generic]]
img_roi: np.ndarray = np.array([1, 2, 3])

app = Flask(__name__)
CORS(app)
counter = 0  # Counter for image filenames

def generate_frames():
    cap = cv2.VideoCapture()
    global img_roi
    while cap.isOpened():
        # read the camera frame
        success, frame = cap.read()
        
        if not success:
            break
        else:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            number_plates = PLATE_CASCADE.detectMultiScale(img_gray, 1.3, 7)

            # Iterate through detected plates
            for (x, y, w, h) in number_plates:
                area = w * h
                if area > MIN_AREA:
                    # Draw rectangle around the plate
                    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR, 2)
                    # Add text label
                    cv2.putText(frame, "License Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)
                    # Show region of interest (ROI)
                    img_roi = frame[y:y + h, x:x + w]
                    

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return "Hello World"


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/save', methods=['POST'])
def save():

    global counter
    global img_roi
    cv2.imwrite(f"/Users/prudhviraj/Documents/CSE443/img/{counter}.png", img_roi)

    image = f"/Users/prudhviraj/Documents/CSE443/img/{counter}.png"
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    extracted_text = pytesseract.image_to_string(image)

    new_text=''
    alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num="0123456789"
    for i in extracted_text:
        if i in alpha or i in num:
            new_text+=i
    print(new_text)
    print("Yes!!!!!")
    collection.insert_one({"number":f"{new_text}"})
    
    counter += 1
    return new_text
if __name__ == '__main__':
    app.run(debug=True)
