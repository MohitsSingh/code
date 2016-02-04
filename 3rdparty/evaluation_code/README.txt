HAND Dataset
==== =======
A. Mittal, A. Zisserman and  P. H. S. Torr


Introduction
------------
This is the evaluation code for Hand Dataset. In [1] we evaluate the performance of our method on all the 'bigger' hand instances (i.e., bounding box area > 1500 sq. pixels) from the test dataset.


Contents
--------
This package contains:
- Bigger hand instances of test dataset which are used for evaluation in [1] (all the images with hand bounding-box annotations in PASCAL format).
- Dummy testing code which could easily be modified for testing with any other method.
- Code to evaluate the performance as per the evaluation criteria.

Let PWD be the directory of the hand-dataset, then the structure of the containts is as follows:

PWD/VOC2007/VOCdevkit/VOCcode: Code from PASCAL development kit.
PWD/VOC2007/VOCdevkit/results: Directory to store the results.
PWD/VOC2007/VOCdetkit/VOC2007: Dataset images with annotations.

run.m: Matlab function to run the detector on the test dataset and evaluate the results . [ THIS IS THE STARTING POINT]
test.m: Function to test the detector on the test dataset.
evaluate_results.m: Scores the bounding boxes using PASCAL development kit
globals.m: Set up global variables used in the code
init.m: Initialize PASCAL development kit.


Evaluation Criteria
---------- --------
The performance is evaluated using average precision (AP) (the area under the Precision Recall curve). A hand detection is considered true or false according to its overlap with the ground-truth bounding box. 
A box is positive if the overlap score is more than 0.5. The overlap ratio is computed between the axis-aligned bounding rectangles around the ground-truth and the detected hand bounding box.

Support
-------
For any query/suggestions, please drop an email to the following addresses:
arpit@robots.ox.ac.uk
az@robots.ox.ac.uk
philiptorr@brookes.ac.uk


References
----------
[1] Hand detection using multiple proposals
A. Mittal, A. Zisserman, P.H.S. Torr
Proceedings of British Machine Vision Conference, 2011.