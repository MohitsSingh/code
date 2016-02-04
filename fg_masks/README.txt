Extracting Foreground Masks towards Object Recognition - Version 2.0 
--------------------------------------------------------------------
Amir Rosenfeld & Daphna Weinshall, October 2012 

https://sites.google.com/site/amirrosenfeld

This package contains the implementation of algorithm presented in
the paper "Extracting Foreground Masks towards Object Recognition".
The code has been totally rewritten and has much less dependencies and has been
simplified a lot.

Running the package should result in segmentations for the PASCAL VOC2011 segmentation 
challenge test-set.


1. Running the Code
-------------------

1.1 Setup
---------

In order to run the code, launch the demo.m script.

This will fail if you don't have one of the prerequisites needed to run the code, so you 
should first download all of them.
The script init.m contains the paths to these pre-requisites, so once you get them,
point the directories in init.m to the right place.

As I use some external packages, remember to cite them if you use my work!

What you'll need is:
1. vlfeat : from vlfeat.org
2. GC-MEX : http://vision.ucla.edu/~brian/gcmex.html
 (there are newer versions but I used this one)
3. some more 3rd party functions.


for any questions, feel free to email me at:

amir.rosenfeld@gmail.com
    or 
amir.rosenfeld@weizmann.ac.il





