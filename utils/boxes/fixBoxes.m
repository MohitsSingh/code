function boxes = fixBoxes(boxes)
boxes(:,[1 3]) = sort(boxes(:,[1 3]),2,'ascend');
boxes(:,[2 4]) = sort(boxes(:,[2 4]),2,'ascend');