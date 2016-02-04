function I_cropped = cropper(I, rect)
%CROPPER cropper(I,rect) crop a rectangular region out of an image and pad
%by zeros if necessary (where rectangle is out of image boundaries)
s = size(I,3);I_cropped = my_arrayCrop(I,[rect([2 1]) 1],[rect([4 3]) s]);
