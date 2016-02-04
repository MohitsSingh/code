function mlixs = EntireMLsOnEdgels(eds, interval)

% which MLs are entirely on edgels of interval ?
%
% Input:
% interval = [e1 e2], with e1<=e2 (sorted in ascending order)
%

used = (eds(3,:)>=interval(1)) & (eds(4,:)<=interval(2));
mlixs = eds(1,used);
