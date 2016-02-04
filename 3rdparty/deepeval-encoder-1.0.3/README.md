ConvNet Feature Computation Package
============================================

Authors:

  * Ken Chatfield, University of Oxford (ken@robots.ox.ac.uk)
  * Karen Simonyan, University of Oxford (karen@robots.ox.ac.uk)

Copyright 2014, all rights reserved.

Release: v1.0.3

Licence: Research only (for now)

Overview
-----------------------------

This package contains a MEX wrapper to compute the ConvNet features described
in the following [BMVC-2014 paper](http://www.robots.ox.ac.uk/~vgg/publications/2014/Chatfield14/chatfield14.pdf):

    K. Chatfield, K. Simonyan, A. Vedaldi, A. Zisserman
    Return of the Devil in the Details: Delving Deep into Convolutional Nets
    British Machine Vision Conference, 2014 (arXiv ref. cs1405.3531)

Please cite the paper if you use the code or the models.

Along with the separately available model files, using the MEX wrapper you can:

 * Compute ConvNet features for a given input image using the main networks described
   in the paper (CNN-F, CNN-M, CNN-S, CNN-M-128 etc.)
 * Set different augmentation and normalisation strategies using just a few lines of code

Further details can be found on the project website, along with details on how to download the accompanying ConvNet model files required to use the MEX wrapper:

 * [http://www.robots.ox.ac.uk/~vgg/research/deep_eval/](http://www.robots.ox.ac.uk/~vgg/research/deep_eval/)

The MEX wrapper is based on our modified version of the publicly available
[Caffe framework](http://caffe.berkeleyvision.org) (forked in Dec 2013).

Requirements
-----------------------------

The following dependencies must be installed to use the MEX wrapper:

* Matlab (to run the MEX file)
* [Intel MKL](https://software.intel.com/en-us/intel-mkl) (free academic licences might be available)
* 64-bit Linux system (sorry - Windows and OSX are not supported at this stage)
* Model files downloaded from project website

Both CPU and GPU computation modes are provided, but by default only CPU computation is
enabled to simplify deployment for environments where a CUDA-compatible GPU is not available.

In order to use GPU mode, in addition to the above requirements CUDA 5.5 or 6.0 must be
installed along with a compatible GPU card. See the next section for instructions on
enabling GPU support.

Activating GPU Support
-----------------------------

In order to compute features using GPU mode, the following additional dependencies must
be met:

* CUDA 5.5/6.0
* NVIDIA GPU card with CUDA compute capability 3.0 or 3.5

To enable GPU computation, replace the default CPU-only computation MEX file
`caffe.mexa64` with either `caffe_60.mexa64` (compiled against CUDA 6.0) or
`caffe_55.mexa64` (compiled against CUDA 5.5) by removing the former file and renaming
one of the two latter files to `caffe.mexa64`.

Usage
-----------------------------

1. Download model files and the mean image from the software page, linked from the
   [project website](http://www.robots.ox.ac.uk/~vgg/research/deep_eval/) and unpack them
   into `models` directory.
2. Edit the Bash file `prep.sh` to adjust the MKL and libstdc++ paths (and CUDA path if
   using GPU computation - the defaults might be fine for you)
3. Edit the demo script `test.m` to:
   a. set the path to the downloaded model
   b. set the path to an input image (a sample image `sample/sample_image.jpg` is used by default)
4. Open a Bash terminal and run `source prep.sh`
5. Launch Matlab from the terminal: `matlab`
6. In Matlab, run the demo script `test.m` to compute features for the sample test image
7. The sample image features can be verified against the reference features, which were
   computed using CNN_M_128 model and are stored in `sample/sample_feat_{backend}.mat`.
   Your features for `sample/sample_image.jpg` should be equal or very close (in terms of
   the L2 distance) to the reference ones.


Source Code Release
-----------------------------

This is an intermediary release of the computation code to allow the ConvNet features
described in the paper to be used immediately. We plan to follow up with a second release
including the source code at a later stage. To be notified when this release is ready,
please [sign up to the project newsletter](http://www.robots.ox.ac.uk/~vgg/research/deep_eval/index.html#newsletter).

Version History
-----------------------------

 * *22 Sept 2014* - **1.0.3** - Small bugfix for mean subtraction in `centre_only_crop` mode
 * *15 Sept 2014* - **1.0.2** - Updated paper reference to conference version
 * *22 July 2014* - **1.0.1** - Fixed class loading to support use in parfor
 * *15 July 2014* - **1.0.0** - Initial public release
