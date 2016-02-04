function dets = transform_dets(dets,boxes_neg,factors)
% bring back detections to image coordinate systems
%   Detailed explanation goes here
for t = 1:length(dets)
    curDets = dets{t};
    curDets(:,1:2) = bsxfun(@plus, curDets(:,1:2)/factors(t) , boxes_neg(t,1:2));
    dets{t} = curDets;
end
