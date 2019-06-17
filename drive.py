import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np
import tensorflow as tf
import cv2



def processImg(img, model):
    shapes =	{
      "vgg": (64, 64),
      "nvidia": (200, 66),
      "dummy": (32, 32)
    }
    img = img[60:135,:,:]
    if(model == "nvidia"):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, shapes[model])
    img = img/255
    return img


def canny(img):
    v = np.median(img)
    sigma = 0.33
    lower = int(max(150, (1.0 - sigma) * v))
    upper = int(max(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


   
# create a Socket.IO server
sio = socketio.Server()
# event sent by the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        input_img = np.asarray(image)
        input_img = processImg(input_img,'vgg')
        input_img = np.array([input_img])
        # Use your model to compute steering and throttle
        print(model.predict(input_img))
        prediction= float(model.predict(input_img)[0][0])
        throttle = 1.0 - speed/30
        print('{} {} {}'.format(prediction, throttle, speed))    #Salimloploi1995!
        send(prediction, throttle)
    else:
        # Edge case
        print('test-1')
        sio.emit('manual', data={}, skip_sid=True)

# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send(0, 0)

# to send steer angle and throttle to the simulator
def send(steer, throttle):
    sio.emit("steer", data={'steering_angle': str(steer), 'throttle': str(throttle)}, skip_sid=True)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# simulator will connect to localhost:4567
if __name__ == '__main__':
    model = tf.keras.models.load_model('./models/model_vgg16_hybrid_rockissue_new_10epchs_64x460x100.h5')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)