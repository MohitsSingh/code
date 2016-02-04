function boxes1 = test

% boxes1 = test
% runs the detector on the test dataset and returns the boxes as cell array

cls = 'hand';
testset = 'hand_test_big';
globals;
init;
year = VOCyear;

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% run detector in each image
try
  load([cachedir cls '_boxes_' testset]);
catch
  % parfor gets confused if we use VOCopts
  opts = VOCopts;
  parfor i = 1:length(ids);
      fprintf('%s: testing: %s %s, %d/%d\n', cls, testset, year, ...
          i, length(ids));
      
      im = imread(sprintf(opts.imgpath, ids{i}));
      
      % sample detector function
      boxes1{i} = sample_detector(im);
      
      % removing the boxes which overlap smaller hands which are not
      % conisdered for evaluation
      boxes1{i} = remove_boxes_ignoreddata(boxes1{i},i);
      
  end    
  save([cachedir cls '_boxes_' testset], ...
       'boxes1');
end

function boxes = sample_detector(im)
% Replace this function with your detector function
% This function generates random number of random bboxes for the image
no_boxes = randi(5,1);
height_im = size(im,1);
width_im = size(im,2);
for i = 1:no_boxes
    x1 = ceil(randi(width_im,1));
    x2 = ceil(randi(width_im,1));
    if(x1 > x2)
        t = x1;
        x1 = x2;
        x2 = t;
    end
    y1 = ceil(randi(height_im,1));
    y2 = ceil(randi(height_im,1));
    if(y1 > y2)
        t = y1;
        y1 = y2;
        y2 = t;
    end
    score = rand(1);
    boxes(i,:) = [x1 y1 (x2-x1+1) (y2-y1+1) score];
end

function boxes = remove_boxes_ignoreddata(boxes,i)
% This function removes the smaller boxes which are not considered for
% evaluation
diff = load('gt_diff_bigandall.mat');%%%%
big = load('gt_test_big.mat');
if(length(diff.gt(i).BB) > 0)
    det_boxes = boxes;
    det_boxes = [det_boxes(:,1) det_boxes(:,2) (det_boxes(:,3)-det_boxes(:,1)+1) (det_boxes(:,4)-det_boxes(:,2)+1)];
    diff_boxes = round(diff.gt(i).BB)';
    diff_boxes = [diff_boxes(:,1) diff_boxes(:,2) (diff_boxes(:,3)-diff_boxes(:,1)+1) (diff_boxes(:,4)-diff_boxes(:,2)+1)];
    isection = rectint(det_boxes,diff_boxes);
    area_det_boxes = repmat(det_boxes(:,3).*det_boxes(:,4),[1 size(isection,2)]);
    overlap = isection ./ area_det_boxes;
    I = find(overlap > 0.35); %0.5
    [row_diff,column_diff] = ind2sub(size(overlap),I);
    
    big_boxes = round(big.gt(i).BB)';
    big_boxes = [big_boxes(:,1) big_boxes(:,2) (big_boxes(:,3)-big_boxes(:,1)+1) (big_boxes(:,4)-big_boxes(:,2)+1)];
    isection = rectint(det_boxes,big_boxes);
    area_det_boxes = repmat(det_boxes(:,3).*det_boxes(:,4),[1 size(isection,2)]);
    overlap = isection ./ area_det_boxes;
    I = find(overlap > 0.5);
    [row_big,column_big] = ind2sub(size(overlap),I);
    
    row_common = intersect(row_diff,row_big);
    
    for j = 1:length(row_diff)
        if(sum(row_common == row_diff(j)) > 0)
            continue;
        end
        boxes(row_diff(j),end) = -Inf;
    end
end
