function m = EdInTheMiddle(e, eds)

% is value e in the middle range [eds(1) eds(2)] ?
% True if it is between the 33% and 66% of the range
%

range = eds(2) - eds(1);
m = ((e - eds(1) >= range*0.33  &  (eds(2) - e) >= range*0.33));
