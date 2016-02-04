function o = IntervalsOverlap(i1, i2)

% overlap between two intervals i1, i2
% (sorted in ascending order)
%

o1 = i1(2)-i2(1)+1;
o2 = i2(2)-i1(1)+1;
o = min([o1 o2 i1(2)-i1(1)+1 i2(2)-i2(1)+1]);
o = max(o,0);
