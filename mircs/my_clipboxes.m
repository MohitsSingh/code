function boxes = my_clipboxes(im, boxes)
% Clips boxes to image boundary.
imy = size(im,1);
imx = size(im,2);
% for i = 1:length(boxes),
%     b = boxes(i,:);
boxes(:,1) = max(boxes(:,1), 1);
boxes(:,2) = max(boxes(:,2), 1);
boxes(:,3) = min(boxes(:,3), imx);
boxes(:,4) = min(boxes(:,4), imy);
%     boxes(i,:) = b;
% end