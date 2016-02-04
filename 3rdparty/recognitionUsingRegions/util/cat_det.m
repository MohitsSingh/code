function detBB = cat_det(detBB,det,id)
% function detBB = cat_det(detBB,det,id)
%
% concatenate det with detBB in the right format.
%
% Copyright @ Chunhui Gu, April 2009

if isempty(detBB),
    detBB = det;
    detBB.id = id*ones(size(detBB.score));
else
    detBB.rect = [detBB.rect; det.rect];
    detBB.score = [detBB.score; det.score];
    detBB.id = [detBB.id; id*ones(size(det.score))];
end;