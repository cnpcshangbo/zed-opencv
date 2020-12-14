import sys
import numpy as np
import pyzed.sl as sl
import cv2
import math
import socket
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				# print(outputFrame is None)
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def point_cloud_format_name():
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing")

def depth_format_name():
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing")

def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)


def process_key_event(zed, key) :
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")

def flaskThread():
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False) 

def main() :
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    ap.add_argument("-s", "--source", type=str, default="cam", help="using 'cam'era or 'svo'<SVO file>?")
    global args
    args = vars(ap.parse_args())

    # start the flask app
    print("running flask.")
    threading.Thread(target=flaskThread).start()
    print("flask app running.")
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if args["source"] != "cam" :
        input_type.set_from_svo_file(args["source"])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', 10000)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth = sl.Mat()
    point_cloud = sl.Mat()

    key = ' '
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    print('connection from ', client_address)

    while key != 113 :
        # Wait for request
        # print('Waiting for request...')
        data = connection.recv(16)
        # print('received {!r}'.format(data))

        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()
            depth_ocv = depth.get_data()
            
            # Detecting wood from depth_image_zed
            # Minimum_value def (y)
            # Get minimum depth value and circle it
            # Input: y colum number
            # Return: Minimum depth value
            n_min = 0
            n=0
            depth_value_min = 2500
            while n < image_size.width :
                x = n;
                y = image_size.height // 2
                depth_value = depth_ocv[y,x]
                if (math.isnan(depth_value)==0) and (depth_value < depth_value_min) and (depth_value > 0):
                    n_min = n
                    depth_value_min = depth_value
                n = n + 1
            # print('Min value is at: ' + str(n_min) + '. Value is: ' + str(depth_value_min) + '.')
            # n_min = 100
            # cv2.circle( depth_image_ocv, ( n_min, image_size.height // 2 ), \
            #        32, ( 0, 0, 255 ), 1, 8 )
            
            # Get points within a threshold and circle them
            # def (minimum_value)
            threshold = depth_value_min + 40
            n = 0
            point_depth = np.array([])
            point_x = np.array([])
            while n < image_size.width :
                x = n
                y = image_size.height // 2
                depth_value =depth_ocv[y,x]
                if math.isnan(depth_value)==0 and depth_value < threshold :
                    point_depth = np.append(point_depth, depth_value)
                    point_x = np.append(point_x, x)
                    # cv2.circle( depth_image_ocv, ( x, image_size.height // 2), \
                    #    16, ( 0, 0, 255 ), 1, 8 )
                n = n + 1
            # print(str(len(point_x)) + 'points are within threshold 40.\n')
            # print('Point within threshold:\n' + str(point_depth))
            girder_center = int(np.mean(point_x))
            # print('Girder ceter at: ' + str(girder_center)) 
            
            # Sending TCP msg
            connection.sendall(girder_center.to_bytes(2,'big'))
    
            # Draw girder center
            cv2.circle( depth_image_ocv, (int(girder_center), image_size.height // \
                    2), 8, (255, 0, 0), 2, 8 )
            # Draw view center
            cv2.circle(depth_image_ocv, (image_size.width //2, image_size.height //2), \
                    8, (0,255,0),2,8)
            # Draw Text depth_value_min
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
  
            # org 
            org = (int(girder_center), image_size.height //2) 
              
            # fontScale 
            fontScale = 1
               
            # Blue color in BGR 
            color = (255, 0, 0) 
              
            # Line thickness of 2 px 
            thickness = 2
               
            # Using cv2.putText() method 
            image = cv2.putText( depth_image_ocv, str(depth_value_min), org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 


            
            # cv2.imshow("Image", image_ocv)
            # cv2.imshow("Depth", depth_image_ocv)
            
            # Send to webserver
            global outputFrame
            outputFrame = depth_image_ocv.copy()
            key = cv2.waitKey(10)

            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()
    # Clean up the connection
    connection.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()
