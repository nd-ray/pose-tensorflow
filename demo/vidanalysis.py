import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib.pyplot as plt
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')

cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)




cap = cv2.VideoCapture('demo/bould.mov')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


i = 1000;

while(cap.isOpened() and i > 0):


    i = i - 1;

    ret, frame = cap.read()

    # Read image from file
    # file_name = "demo/climber.jpg"
    # image = imread(file_name, mode='RGB')
    image = frame

    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

    detections = extract_detections(cfg, scmap, locref, pairwise_diff)
    unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
    person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

    img = np.copy(image)

    visim_multi = img.copy()
    plt.clf()
    fig = plt.imshow(visim_multi)
    draw_multi.draw(visim_multi, dataset, person_conf_multi)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    # plt.show()
    plt.savefig('testplot.png')
    out.write(imread('testplot.png', mode='RGB'))
    # visualize.waitforbuttonpress()

    #fig = plt.figure()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('frame',visim_multi)

    #fig.canvas.draw()
    # convert canvas to image
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    #                     sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    # # img is rgb, convert to opencv's default bgr
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #
    # # display image with opencv or any operation you like
    # cv2.imshow("plot", imread('testplot.png', mode='RGB'))



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



