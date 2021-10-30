import sys
import argparse
from yolo import YOLO, detect_video
#from yolo_1 import YOLO_1, detect_video_1
from PIL import Image
import cv2
import numpy
import multiprocessing 

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            # r_image.show()
            opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite('pictures/test_result.png',opencvImage)
    yolo.close_session()

FLAGS = None

     
if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./phase3.mp4',
        help = "Video input path"
        
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="./final_Demo.mp4",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    i=YOLO(**vars(FLAGS))
    j=YOLO(**vars(FLAGS))
    k=YOLO(**vars(FLAGS))
    l=YOLO(**vars(FLAGS))
    #print(vars(FLAGS))

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            #print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
            detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        #print('FLAGS.output: ',FLAGS.output)
        #detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
        
        
        detect_video(i,j ,k, l,FLAGS.input, FLAGS.output)
        
           
        #thread=multiprocessing.Process(target=detect_video(YOLO(**vars(FLAGS)),FLAGS.input, FLAGS.output))
        #thread.start()

        
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
        
        
	
