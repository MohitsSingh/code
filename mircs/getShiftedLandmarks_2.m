function [xy_c,mouth_poly,face_poly] = getShiftedLandmarks_2(landmarks,ref_box)
% Returns the coordinates of the facial landmarks outline of the area of
% the face, which were extracted from the image cropped to the bounding box
% of the person. 
if (nargin < 2)
    ref_box = [0 0];
end


% xy = landmarks.xy;
% xy_c = bsxfun(@minus,boxCenters(xy),ref_box([1 2]));
xy_c = cellfun2(@(x) mean(x,1),landmarks.polys);
xy_c = cat(1,xy_c{:});

% xy_c = boxCenters(xy);

if (size(xy_c,1)==68)
    outline_ = [68:-1:61 52:60 16:20 31:-1:27];
    inner_lips = [36 37 38 42 42 45 47 49 36];
    outer_lips = [35 34 33 32 39 40 41 44 46 51 48 50 35];
    mouth_corner = [35 41];
else
    outline_ = [6:-1:1 16 25 27 22 28:39 15:-1:12];
    inner_lips = [25 24 26 23 27];
    outer_lips = [16:22];
    mouth_corner = 19;
end

mouth_poly = xy_c(outer_lips,:);
face_poly = xy_c(outline_,:);