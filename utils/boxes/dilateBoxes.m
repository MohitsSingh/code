function boxes = dilateBoxes(boxes,boxMargin)
% add a constant to the size of each boxes from each side.
boxes(:,1:2) = boxes(:,1:2)-boxMargin;
boxes(:,3:4) = boxes(:,3:4)+boxMargin;

% fix it.
boxes = fixBoxes(boxes);



end