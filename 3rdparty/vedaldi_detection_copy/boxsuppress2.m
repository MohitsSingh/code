function keep = boxsuppress2(boxes, scores, threshold)
% BOXSUPPRESS Box non-maxima suprression
%   KEEP = BOXSUPPRESS(BOXES, SCORES, THRESHOLD)

% remove any empty box (xmax < xmin or ymax < ymin)
scores(any([-1 0 1 0 ; 0 -1 0 1] * boxes < 0)) = -inf ;
opts = struct('pascalFormat', true);
keep = false(1, size(boxes,2)) ;
while true
  [score, best] = max(scores) ;
  if score == -inf, break ; end
  keep(best) = true ;
  remove = boxinclusion2(boxes(:,best), boxes, opts) >= threshold ;
  scores(remove) = -inf ;
  scores(best) = -inf ; % `best` is not in `remove` if threshold > 1
end
