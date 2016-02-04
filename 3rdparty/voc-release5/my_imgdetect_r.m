function [boxes, boxes_r, bboxes, info] = my_imgdetect_r(input, model, thresh, fastflag, bbox, overlap)

% Wrapper that computes detections in the input image.
%
% input    input image
% model    object model
% thresh   detection score threshold
% bbox     ground truth bounding box
% overlap  overlap requirement

if nargin < 5
    bbox = [];
end

if nargin < 6
    overlap = 0;
end

[h,w,d] = size(input);
diagonal = round(sqrt(h^2 + w^2));
ypadding = round(0.5*(diagonal-h));
xpadding = round(0.5*(diagonal-w));
input = padarray(input,[ypadding xpadding]);
[h,w,d] = size(input);

bboxes = [];
boxes = [];
boxes_r = [];
disp('Detecting at 36 different rotations');
for angle = -170:10:180
% for angle = 0
%     disp((angle+170+10)/10);
    rinput = imrotate(input,angle,'bilinear','crop');
    
    % we assume color images
    rinput = color(rinput);
    
    % get the feature pyramid
%     pyra = featpyramid(rinput, model);
    
%     if(fastflag)
%         % The cascade was trained on all of the hand training dataset
%         thresh = -1;
%         [boxes_a, bboxes_a] = cascade_detect(pyra, model, thresh);  % faster cascaded version
%     else
%         [boxes_a, bboxes_a, info_a] = gdetect(pyra, model, thresh, bbox, overlap);  % slower version
%     end
%    
    [boxes_a, bboxes_a] = imgdetect(rinput, model, thresh);
    %[boxes_a, bboxes_a, info_a] = imgd(pyra, model, thresh, bbox, overlap);  % slower version

    [boxes_ra, boxesr_ra] = antirotate_r(boxes_a,angle,h,w,ypadding,xpadding);
    
    boxes = [boxes; esvm_nms( boxes_ra , .7)];
    bboxes = boxes;
    boxes_r = [boxes_r; boxesr_ra];
end
info = [];

function [r_matrix, rr_matrix] = antirotate_r(matrix,angle,height,width,ypad,xpad)
r_matrix = zeros(size(matrix));
rr_matrix = zeros(size(matrix,1),10);
for i = 1:size(matrix,1)
    for j = 1:floor(size(matrix,2)/4)
        x1 = matrix(i,(j-1)*4+1);
        y1 = matrix(i,(j-1)*4+2);
        x2 = matrix(i,(j-1)*4+3);
        y2 = matrix(i,(j-1)*4+4);
        [p1_x,p1_y] = ptRotate(x1,y1,-angle,height,width);
        [p2_x,p2_y] = ptRotate(x2,y1,-angle,height,width);
        [p3_x,p3_y] = ptRotate(x2,y2,-angle,height,width);
        [p4_x,p4_y] = ptRotate(x1,y2,-angle,height,width);
        r_matrix(i,(j-1)*4+1) = min([p1_x p2_x p3_x p4_x]) - xpad;
        r_matrix(i,(j-1)*4+2) = min([p1_y p2_y p3_y p4_y]) - ypad;
        r_matrix(i,(j-1)*4+3) = max([p1_x p2_x p3_x p4_x]) - xpad;
        r_matrix(i,(j-1)*4+4) = max([p1_y p2_y p3_y p4_y]) - ypad;
        
        rr_matrix(i,(j-1)*8+1) = p1_x - xpad;
        rr_matrix(i,(j-1)*8+2) = p1_y - ypad;
        rr_matrix(i,(j-1)*8+3) = p2_x - xpad;
        rr_matrix(i,(j-1)*8+4) = p2_y - ypad;
        rr_matrix(i,(j-1)*8+5) = p3_x - xpad;
        rr_matrix(i,(j-1)*8+6) = p3_y - ypad;
        rr_matrix(i,(j-1)*8+7) = p4_x - xpad;
        rr_matrix(i,(j-1)*8+8) = p4_y - ypad;
    end
    r_matrix(i,end-mod(size(matrix,2),4)+1:end) = matrix(i,end-mod(size(matrix,2),4)+1:end);
    rr_matrix(i,(j-1)*8+9:(j-1)*8+10) = matrix(i,end-mod(size(matrix,2),4)+1:end);
end


function r_matrix = antirotate(matrix,angle,height,width)
r_matrix = zeros(size(matrix));
for i = 1:size(matrix,1)
    for j = 1:floor(size(matrix,2)/4)
        x1 = matrix(i,(j-1)*4+1);
        y1 = matrix(i,(j-1)*4+2);
        x2 = matrix(i,(j-1)*4+3);
        y2 = matrix(i,(j-1)*4+4);
        [p1_x,p1_y] = ptRotate(x1,y1,-angle,height,width);
        [p2_x,p2_y] = ptRotate(x2,y1,-angle,height,width);
        [p3_x,p3_y] = ptRotate(x2,y2,-angle,height,width);
        [p4_x,p4_y] = ptRotate(x1,y2,-angle,height,width);
        r_matrix(i,(j-1)*4+1) = min([p1_x p2_x p3_x p4_x]);
        r_matrix(i,(j-1)*4+2) = min([p1_y p2_y p3_y p4_y]);
        r_matrix(i,(j-1)*4+3) = max([p1_x p2_x p3_x p4_x]);
        r_matrix(i,(j-1)*4+4) = max([p1_y p2_y p3_y p4_y]);
    end
    r_matrix(i,end-mod(size(matrix,2),4)+1:end) = matrix(i,end-mod(size(matrix,2),4)+1:end);
end

function [newx newy] = ptRotate(x,y,angle,h,w)
angle = angle*pi/180;
newy = -x*sin(angle)+y*cos(angle)+h/2+w/2*sin(angle)-h/2*cos(angle);
newx = x*cos(angle)+y*sin(angle)+w/2-w/2*cos(angle)-h/2*sin(angle);
newy = round(newy); newx = round(newx);
if(newx < 1)
	newx = 1;
end
if(newx > w)
	newx = w;
end
if(newy < 1)
	newy = 1;
end
if(newy > h)
	newy = h;
end
