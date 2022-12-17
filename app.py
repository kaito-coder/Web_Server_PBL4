import cv2
import os
import requests

from flask import Flask, render_template,Response, redirect, url_for, request
from modules import YoloV5

app=Flask(__name__,  static_url_path='/static', static_folder='static')
camera = None

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        #camera.release()
        #cv2.destroyAllWindows() 
        yield(b'--frame\r\n'
                   b'Con6tent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_frame():
    global camera
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            cv2.imwrite("static/test.jpeg", frame)
            break


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take-pic')
def take_pic():
    get_frame()
    return redirect(url_for('predict'))

@app.route('/predict')
def predict():
    yolov5 = YoloV5(weight_path = "weights/best.pt", image_path = "static/test.jpeg")
    final_results = yolov5.predict_labels()
    yolov5.inference()
    return render_template('inference.html', label=final_results[0][0], accuracy=(final_results[0][1] * 100), image='static/inference.jpeg')




""" 
    Server request to arduino
"""
@app.route('/helloesp')
def helloHandler():
    return 'Hello ESP8266, from Flask'

@app.route('/stop')
def get_data1():
    return requests.get('http://192.168.1.208/dung').content

@app.route('/line')
def get_data2():
        return  requests.get('http://192.168.1.208/line').content
                  
@app.route('/left')
def get_data3():
    return requests.get('http://192.168.1.208/trai').content

@app.route('/right')
def get_data4():
    return requests.get('http://192.168.1.208/phai').content









if __name__=="__main__":
    app.run(debug=True, port=2404, host='0.0.0.0')
