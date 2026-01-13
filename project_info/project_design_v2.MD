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
