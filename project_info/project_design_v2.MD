# Project: Automated Volleyball Match Analysis Pipeline

### ** 1. improved RF training pipeline **
currently the training pipelines get extracted features, and shows accuracy before smoothing
in version 2 the pipeline will start at the detections and show accuracy after smoothing
this will allow to easily see the improvements of modifications in the pipeline

### ** 2. change feature extraction strategy **
currently there are a lot of irrelevant features, and the window is centered on the predicted frame
in version two I will experiment with different features(using what I learned from version 1)
and try to play with the window to give the context direction with things like, a before and after window
and features that have directionality for example (start track cx,cy)  

### **3. use ball size **
the a yolo model predicts a box around the ball. currently the center of the box 
is the only data used in the next stages of pipeline, using the size could improve 
the accuracy of certain things(eg a track of the ball becomes a hand might be prevented thanks to size)

### **4. calibration stage **
this is a new stage of the pipeline.
the goal of it to improve the understanding of the video and not
just the tracks
it is still to be tbd, but the goal is that the model will know
more things for example what is close to it vs is far
where is the ball likely to enter at the start of rally
and generalize all the features, based on what was seen in the video
vs hard coded number
for example depending on the angle of film, we may want different threshold for things


### **Goal for v2** ###
the main goal for v2 is to improve accuracy to handle the serve error skip
(improve the accuracy of the RF to ~99%) 

