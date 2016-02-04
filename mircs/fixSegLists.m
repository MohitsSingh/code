function seglists = fixSegLists(seglists,coord)
% seglists = fixSegLists(seglists) fix each sequence of segments so
% lefmost segment's left point is first in the sequence.
if (nargin < 2)
    coord = 'x';
end
if (coord == 'x')
    ic = 2;
else
    ic = 4;
end
for k = 1:length(seglists)
    curList = seglists{k};
    firstX = curList(1,ic);
    lastX = curList(end,ic+2);
    if (firstX > lastX) % switch direction of each segment and their order.
        curList = curList(:,[3 4 1 2]);
        curList = flipud(curList);
        seglists{k} = curList;
    end
    %otherwise, there's nothing to fix here.
    
end
end