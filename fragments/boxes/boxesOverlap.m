function [ overlaps ] = boxesOverlap( bboxes,bboxes2)
%BOXESOVERLAP Computes overlap (intersection / union) between each pair in
%bboxes. Returns a matrix whose i,j element is the intersection between
%element i and element j in bboxes, or element i in bboxes and element j in
%bboxes22 if bboxes2 is specified;

if (nargin == 2)
    nBoxes = size(bboxes,1);
    nBoxes2 = size(bboxes2,1);
    overlaps = zeros(nBoxes,nBoxes2);
    
    for iBox = 1:nBoxes2
        q = repmat(bboxes2(iBox,:),nBoxes,1);
        boxes_i = BoxIntersection(q,bboxes);
        has_intersection = boxes_i(:,1) ~= -1;
        [~, ~, bi] = BoxSize(boxes_i(has_intersection,:));
        [~,~,b0] = BoxSize(q(has_intersection,:));
        [~,~,b1] = BoxSize(bboxes(has_intersection,:));
        bu = b0+b1-bi;
        bo = bi./bu;
        overlaps(has_intersection,iBox) = bo;        
    end
else
    nBoxes = size(bboxes,1);    
    overlaps = zeros(nBoxes);
    %     tic
    for iibox = 1:nBoxes
        bbox_i = repmat(bboxes(iibox,:),nBoxes-iibox,1);
        boxes_i = BoxIntersection(bbox_i,bboxes(iibox+1:end,:));
        has_intersection = boxes_i(:,1) ~= -1;
        [~, ~, bi] = BoxSize(boxes_i(has_intersection,:));
        
        %%%
        [~,~,b0] = BoxSize(bbox_i);
        [~,~,b1] = BoxSize(bboxes(iibox+1:end,:));
        b0 = b0(has_intersection);
        b1 = b1(has_intersection);                
        %%%        
        %[~, ~, bu] = BoxSize(BoxUnion(bbox_i,bboxes(iibox+1:end,:)));
        bu = b0+b1-bi;
        L = overlaps(iibox,iibox+1:end);
        L(has_intersection) = bi./bu;
        overlaps(iibox,iibox+1:end) = L;
    end
end