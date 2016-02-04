---------------------------------------------------- 
Pose Estimation results
---------------------------------------------------- 
This package includes pose estimation results on the LSP, Extended LSP and FLIC dataset of Chen and Yuille, NIPS'14:
@InProceedings{Chen_NIPS14,
 title        = {Articulated Pose Estimation by a Graphical Model with Image Dependent Pairwise Relations},
 author       = {Xianjie Chen and Alan Yuille},
 booktitle    = {Advances in Neural Information Processing Systems (NIPS)},
 year         = {2014},
}

---------------------------------------------------- 
Pose Estimation Evaluation Code
---------------------------------------------------- 
The results are evaluated by strict Percentage of Correct Parts (PCP), and the curve of Percentage of Detected Joints (PDJ).

----------------------------------------------------  
strict PCP
---------------------------------------------------- 
Please refer to Ferrari et al. CVPR’08 for the PCP definition:
@inproceedings{ferrari2008progressive,
  title={Progressive Search Space Reduction for Human Pose Estimation},
  author={Ferrari, Vittorio and Marin-Jimenez, Manual and Zisserman, Andrew},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2008},
}

Also please see the webpage http://www.robots.ox.ac.uk/~vgg/data/stickmen/ and Pishchulin et al. CVPR'12 
for a clarification of the PCP evaluation criterion:
@inproceedings{pishchulin2012articulated,
  title={Articulated people detection and pose estimation: Reshaping the future},
  author={Pishchulin, Leonid and Jain, Arjun and Andriluka, Mykhaylo and Thormahlen, T and Schiele, Bernt},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2012},
}
We use the stricter version of PCP.

----------------------------------------------------  
PDJ
---------------------------------------------------- 
Please refer to Sapp and Taskar CVPR’13 for the PDJ definition:
@inproceedings{sapp2013modec,
  title={MODEC: Multimodal decomposable models for human pose estimation},
  author={Sapp, Ben and Taskar, Ben},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2013},
}
