function sums = sum_boxes(I,boxes)
% use integral image to sum many boxes in I (2 dim image);

M = vl_imintegral(padarray(I,[1 1],0,'pre'));
xmin = boxes(:,1);ymin = boxes(:,2);
xmax = boxes(:,3)+1;ymax = boxes(:,4)+1;
sz = size(M);
sums = M(sub2ind2(sz,[ymax,xmax]))-...
    M(sub2ind2(sz,[ymax,xmin]))-...
    M(sub2ind2(sz,[ymin,xmax]))+...
    M(sub2ind2(sz,[ymin,xmin]));

end