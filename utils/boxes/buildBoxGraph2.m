function G = buildBoxGraph2(boxes1,boxes2,boxMargin)

if nargin < 2
    boxes2 = boxes1;
end
if nargin < 3
    boxMargin = 3;
end

boxes_bigger1 = dilateBoxes(boxes1,boxMargin);
boxes_smaller1 = dilateBoxes(boxes1,-boxMargin);
boxes_bigger2 = dilateBoxes(boxes2,boxMargin);
boxes_smaller2 = dilateBoxes(boxes2,-boxMargin);
% profile on
ovp = boxesOverlap(boxes1,boxes2) > 0;
%ovp = boxesHaveIntersection2(boxes1,boxes2);
% profile viewer
%ovp_big = boxesHaveIntersection(boxes_bigger1,boxes_bigger2);
ovp_big = boxesOverlap(boxes_bigger1,boxes_bigger2) > 0;
ovp_small = boxesOverlap(boxes_smaller1,boxes_smaller2) > 0;
G = (ovp_big & ~ovp) | (ovp & ~ovp_small);