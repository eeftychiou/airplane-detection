##Basic execution 

python detect.py --cascade classifier/cascade.xml --video samplevid/plane.mp4 --showHaarDetections



##Help File

usage: detect.py [-h] --cascade CASCADE --video VIDEO [--savedetections]
                 [--savemotiondetections] [--overlapratio OVERLAPRATIO]
                 [--savediffs] [--showHaarDetections] [--showMotionDetections]
                 [--UpdateLists] [--EnableMotionDetect] [--EnableOutStream]
                 [--SaveOutStream]

*Process options*

```
optional arguments:
  -h, --help            show this help message and exit
  --cascade CASCADE     define cascade classifier
  --video VIDEO         define video
  --savedetections      save Haar Detections
  --savemotiondetections
                        Save motion detections
  --overlapratio OVERLAPRATIO
                        Overlap Ratio 0-1
  --savediffs           Save detection differences. That is save all images
                        not overlaping
  --showHaarDetections  Show Haar Detections in the window
  --showMotionDetections
                        Show Motion Detections in the window
  --UpdateLists         Updates the positive and negative lists
  --EnableMotionDetect  Enables Motion Detection
  --EnableOutStream     Enables Streaming to pipe
  --SaveOutStream       Saves Image Stream for video Creation
```
