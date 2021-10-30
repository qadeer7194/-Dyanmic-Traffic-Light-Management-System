# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import cv2
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from sort import *
import led
tf.compat.v1.disable_eager_execution()
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

class YOLO(object):
    _defaults = {
         "model_path": 'model_data/yolo.h5', 
         #"classes_path": 'model_data/coco_classes.txt',                  # 0 original yolo model
        # "model_path": 'model_data/vehiclemodelfirst.h5',         # 1 to test the derived model for coco-dataset
        
       # "model_path": 'model_data/yolo_weights1000.h5',           # 2-1) to test the coco_dataset_derived_model
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',         # 2-2) to test the coco_dataset_derived_model
        
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.tracker1 = Sort()
        self.counter1=0
        self.memory1 = {}
        self.line = [(30, 615), (528, 615)]
        self.previous1 = {}
        self.tracker2 = Sort()
        self.counter2=0
        self.memory2 = {}
        self.previous2 = {}
        self.tracker3 = Sort()
        self.counter3=0
        self.count1=0
        self.memory3 = {}
        self.previous3 = {}

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
  
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            #print('output_shape = %d' %(self.yolo_model.layers[-1].output_shape[-1]))
            #print('num_anchors = %d' % num_anchors)
            #print('len = %d' %(len(self.yolo_model.output) * (num_classes + 5)))
            #print('len_output = %d' %(len(self.yolo_model.output)))
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors/len(self.yolo_model.output) * (num_classes + 5), 'Mismatch between model and given anchor and class sizes'

        #print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        global count1 
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        img=image
        image = np.array(image)
        oH,oW = image.shape[:2]
        ROI_YMAX = int(round(oH * 0.75))  # Bottom quarter = finish line
        ROI = int(round(oH * 0.71))  # Bottom quarter = finish line
        line = [(0, ROI), (oW, ROI)]
        #print(image.shape)

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
        #idxs = cv2.dnn.NMSBoxes(out_boxes, out_scores,0.4 ,0.3)
        #print("here is out_scores:",out_scores)
        #print("here is out_boxes:",out_boxes)
       # print("here is out_classes:",out_classes)
        dets1 = []
        dets2 = []
        dets3 = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(img.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (top, left), (bottom, right))
            #cv2.rectangle(image, (left, top), (right,bottom), (0,255,0), 2)
            if predicted_class == "bicycle" or predicted_class == "motorbike":
                    dets1.append([left, top, right, bottom, out_scores[i]])
            if predicted_class == "truck" or predicted_class == "bus":
                    dets2.append([left, top, right, bottom, out_scores[i]])
            if predicted_class == "car":
                    dets3.append([left, top, right, bottom, out_scores[i]])

        '''idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.5)
       # print("idx:",idx)
	
        boxs = []
        clas = []
        score = []
        if len(idxs) > 0:
	 	# loop over the indexes we are keeping
                for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        boxs.append([x, y, w, h])
                        clas.append([classes[i]])
                        score.append([scores[i]])'''


      #  print("here is dets1:",dets1)
       # print("here is dets2:",dets2)
        #print("here is dets3:",dets3)


        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


        dets1 = np.asarray(dets1)
        dets2 = np.asarray(dets2)
        dets3 = np.asarray(dets3)


        tracks1 = self.tracker1.update(dets1)
        tracks2 = self.tracker2.update(dets2)
        tracks3 = self.tracker3.update(dets3)

        boxex1 = []
        indexIDs1 = []
        c1 = []
        self.previous1 = self.memory1.copy()
        self.memory1 = {}

        boxex2 = []
        indexIDs2 = []
        c2 = []
        self.previous2 = self.memory2.copy()
        self.memory2 = {}

        boxex3 = []
        indexIDs3 = []
        c3 = []
        self.previous3 = self.memory3.copy()
        self.memory3 = {}

        for track in tracks1:
                boxex1.append([track[0], track[1], track[2], track[3]])
                indexIDs1.append(int(track[4]))

                self.memory1[indexIDs1[-1]] = boxex1[-1]
        #print("here is memory:",self.memory)

        for track in tracks2:
                boxex2.append([track[0], track[1], track[2], track[3]])
                indexIDs2.append(int(track[4]))
                self.memory2[indexIDs2[-1]] = boxex2[-1]
        #print("here is memory:",self.memory)

        for track in tracks3:
                boxex3.append([track[0], track[1], track[2], track[3]])
                indexIDs3.append(int(track[4]))
                self.memory3[indexIDs3[-1]] = boxex3[-1]
        #print("here is memory:",self.memory)

        if len(boxex1) > 0:
                i = int(0)
                for box in boxex1:
			# extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        color = (0,255,255)
                        
                        cv2.rectangle(image, (x, y), (w, h), (0,255,0), 2)

                        if indexIDs1[i] in self.previous1:
                                previous_box = self.previous1[indexIDs1[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                cv2.line(image, p0, p1, (255,0,0), 3)

                                if self.intersect(p0, p1, line[0], line[1]):
                                        self.counter1 += 1
                                        #print("actual coutner is here:",self.counter)

			#text = "{}: {:.4f}".format(LABELS[out_classes[i]], out_scores[i])
                        text = "{}".format(indexIDs1[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        #self.count1+=1
                        i += 1
                        
        if len(boxex2) > 0:
                i = int(0)
                for box in boxex2:
			# extract the bounding box coordinates
                        #c(x, y) = (int(box[0]), int(bophoto-1552519507-da3b142c6e3d.jpegx[1]))
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        color = (0,255,255)
                        
                        cv2.rectangle(image, (x, y), (w, h), (0,255,0), 2)

                        if indexIDs2[i] in self.previous2:
                                previous_box = self.previous2[indexIDs2[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                cv2.line(image, p0, p1, (255,0,0), 3)

                                if self.intersect(p0, p1, line[0], line[1]):
                                        self.counter2 += 1
                                        #print("actual coutner is here:",self.counter)

			#text = "{}: {:.4f}".format(LABELS[out_classes[i]], out_scores[i])
                        text = "{}".format(indexIDs2[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        #self.count1+=1
                        i += 1
                        
        if len(boxex3) > 0:
                i = int(0)
                for box in boxex3:
			# extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        color = (0,255,255)
                        
                        cv2.rectangle(image, (x, y), (w, h), (0,255,0), 2)

                        if indexIDs3[i] in self.previous3:
                                previous_box = self.previous3[indexIDs3[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                cv2.line(image, p0, p1, (255,0,0), 3)

                                if self.intersect(p0, p1, line[0], line[1]):
                                        self.counter3 += 1
                                        #print("actual coutner is here:",self.counter)

			#text = "{}: {:.4f}".format(LABELS[out_classes[i]], out_scores[i])
                        text = "{}".format(indexIDs3[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        #self.count1+=1
                        i += 1
                        
                        
                         # draw line
        # OpenCV colors are (B, G, R) tuples -- RGB in reverse
        WHITE = (255, 255, 255)
        YELLOW = (66, 244, 238)
        GREEN = (80, 220, 60)
        LIGHT_CYAN = (255, 255, 224)
        DARK_BLUE = (139, 0, 0)
        GRAY = (128, 128, 128)    
    
        # Add finish line overlay/line
        overlay = image.copy()

        # Shade region of interest (ROI). We're really just using the top line.
        cv2.rectangle(overlay,
                      (0, ROI_YMAX),
                      (oW, oH), DARK_BLUE, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        color = (80, 220, 60)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        thickness = 1

        cv2.line(image, (0, ROI_YMAX), (oW, ROI_YMAX), YELLOW, 4, cv2.LINE_AA)
        cv2.line(image, (oW//3, ROI_YMAX - 20), (oW//3, ROI_YMAX + 20), LIGHT_CYAN, 4, cv2.LINE_AA)
        cv2.line(image, (oW//3 *2, ROI_YMAX - 20), (oW//3 *2, ROI_YMAX + 20), LIGHT_CYAN, 4, cv2.LINE_AA)

#cv2.line(image, (image.shape[1]//3 *3, ROI_YMAX - 20), (image.shape[1]//3 *3, ROI_YMAX + 20), LIGHT_CYAN, 4, cv2.LINE_AA)

        #cv2.putText(image, "Ambulance:", (((oW//3)//2)-12, ROI_YMAX + 50), fontface, 1, LIGHT_CYAN, 2, cv2.LINE_AA)
        cv2.putText(image, "Car:", ((((oW//3)//2)*3)-12, ROI_YMAX + 50), fontface, 1, GREEN, 2, cv2.LINE_AA)
        #cv2.putText(image, "TWO WHEEL:", ((((oW//3)//2)*5)-80, ROI_YMAX + 50), fontface, 1, LIGHT_CYAN, 2, cv2.LINE_AA)

        #cv2.putText(image, str(self.counter2), (((oW//3)//2)-5, ROI_YMAX + 100), fontface, 1, GREEN, 2, cv2.LINE_AA)
        cv2.putText(image, str(self.counter3), ((((oW//3)//2)*3)-5, ROI_YMAX + 100), fontface, 1, GREEN, 2, cv2.LINE_AA)
        #cv2.putText(image, str(self.counter1), ((((oW//3)//2)*5)-5, ROI_YMAX + 100), fontface, 1, GREEN, 2, cv2.LINE_AA)
        self.count1=self.counter2+self.counter3+self.counter1
        
        #cv2.line(image, self.line[0], self.line[1],(0, 255, 255), 5)

                        # draw counter
        #cv2.putText(image, str(self.counter1)+","+str(self.counter2)+","+str(self.counter3), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
                        # counter += 1

        return image
      #counting return  
    def counting(self):
        return self.count1  
        	
         		
         	 
	  
    def close_session(self):
        self.sess.close()

      	    
def detect_video(yolo,yolo1,yolo2,yolo3,video_path, output_path="final_demo.mp4"):
    #initilize the camera
    
    vid = cv2.VideoCapture(2)
    vid1 = cv2.VideoCapture(6)	
    vid2  = cv2.VideoCapture(4)
    #vid3  = cv2.VideoCapture(4)
    #vid1  = cv2.VideoCapture(4)
    #vid2  = cv2.VideoCapture(2)
    #os.makedirs(out_image_folder, exist_ok=True)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam1 or video") 
      #second
    if not vid1.isOpened():
        raise IOError("Couldn't open webcam2 or video") 
    if not vid2.isOpened():
        raise IOError("Couldn't open webca3 or video") 
    #if not vid3.isOpened():
     #   raise IOError("Couldn't open webca4 or video") 
              
    #ist
    video_FourCC    = cv2.VideoWriter_fourcc(*"XVID")
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
     #second
    video_FourCC1    = cv2.VideoWriter_fourcc(*"XVID") 
    video_fps1       = vid1.get(cv2.CAP_PROP_FPS)
    video_size1      = (int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
 
#third 
    video_FourCC2    = cv2.VideoWriter_fourcc(*"XVID")
    video_fps2       = vid2.get(cv2.CAP_PROP_FPS)
    video_size2      = (int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
#forth 
    #video_FourCC2    = cv2.VideoWriter_fourcc(*"XVID")
    #video_fps3       = vid3.get(cv2.CAP_PROP_FPS)
    #video_size3      = (int(vid3.get(cv2.CAP_PROP_FRAME_WIDTH)),
     #                   int(vid3.get(cv2.CAP_PROP_FRAME_HEIGHT))) 

 
#ist                       
    isOutput = True if output_path != "" else False
    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
     #2nd   
    isOutput = True if output_path != "" else False
    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps1), type(video_size1))
    out1 = cv2.VideoWriter(output_path, video_FourCC, video_fps1, video_size1)
#3rd
    isOutput = True if output_path != "" else False
    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps1), type(video_size1))
    out2 = cv2.VideoWriter(output_path, video_FourCC, video_fps2, video_size2)
#3rd
    isOutput = True if output_path != "" else False
    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps2), type(video_size1))
    out3 = cv2.VideoWriter(output_path, video_FourCC, video_fps2, video_size2)

    while True:
        #ist
        #count1=0
        #read the frames
        return_value, frame = vid.read()
        #2nd
        rev,frame1=vid1.read()    
        #3rd
        rev1,frame2=vid2.read()
   #     rev2,frame3=vid3.read()
        #ist
        if not return_value:
            break
        #2nd
        if not rev:
            break
        #3rd
        if not rev1:
            break
       # if not rev2:
        #    break

        # print('frame: ', frame)
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)#detecting car
        result = np.asarray(image)
        print('camera1',yolo.counting())
        c1=yolo.counting()
        #seond
        image1 = Image.fromarray(frame1)
        image1 = yolo1.detect_image(image1)
        result1 = np.asarray(image1)
        print('camera2',yolo1.counting())
        c2=yolo1.counting()
        #3rd
        image2 = Image.fromarray(frame2)
        image2 = yolo2.detect_image(image2)
        result2 = np.asarray(image2)
        print('camera3',yolo2.counting())
        c3=yolo2.counting()
        
        #image3 = Image.fromarray(frame3)
        #image3 = yolo2.detect_image(image3)
        #result3 = np.asarray(image3)
        #print('camera4',yolo3.counting())
        #c4=yolo3.counting()
        led.switch(c3,c2,c1,14) #call the switch light function 


        cv2.namedWindow("D", cv2.WINDOW_NORMAL)#show window
        cv2.imshow("D", result)

        cv2.namedWindow("B", cv2.WINDOW_NORMAL)
        cv2.imshow("B",result1)

        cv2.namedWindow("A", cv2.WINDOW_NORMAL)
        cv2.imshow("A",result2)

        #cv2.namedWindow("4th", cv2.WINDOW_NORMAL)
        #cv2.imshow("4th",result3)
        

        #count += 1
       # print('count: ', count)
        # cv2.imwrite("test_data/output/result_%d.jpg" % count, result)
        #cv2.imwrite(out_image_folder+"/result_%d.jpg" % count, result)

        out.write(result)
        out1.write(result1)
        out2.write(result2)
  #      out3.write(result3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #ist
    out.release()
    #2nd
    out1.release()
    #3rd
    out2.release()
    #out3.release()
   # out3.relase()
    vid.release()
    vid1.release()
    vid2.release()
    #vid3.relase()
    yolo.close_session()
    
     	
    
	
    



# test
# cv2.imwrite("frame_1.jpg", frame)
