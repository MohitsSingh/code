function bb = imbox(I)
% return the bounding box corresponding to the borders of the image.
bb = [1 1 fliplr(size2(I))];
