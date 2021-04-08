# load the trained model and label binarizer from disk
import pickle
import cv2
import argparse
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np

cl= argparse.ArgumentParser()
cl.add_argument("-i", "--input",required=True,help="input video to classify")
cl.add_argument("-m", "--model",required=True,help="path of saved trained model to use")
cl.add_argument("-l","--labels",required=True,help="path to the pickled labels file")
cl.add_argument("-q","--queue",required=False,help="#of frames to average over")
args = vars(cl.parse_args())


model = load_model(args["model"])
lb = pickle.loads(open(args["labels"], "rb").read())

Q = deque(maxlen=int(args["queue"]))

#vid='/home/emmanouela/Documents/mosAIc.gr/2DCNN/original_models/videos/football.mp4'
vid=args["input"]
# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(vid)

# loop over frames from the video file stream
ff=0
while True:
# read the next frame from the file
    (grabbed, frame) = vs.read()
    ff+=1
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
        
    #clone the output frame, then convert it from BGR to RGB
    #ordering, resize the frame 
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    
    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    
    
    # draw the activity on the output frame
    text = "{}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 5)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


vs.release()
