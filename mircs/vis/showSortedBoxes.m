function showSortedBoxes(I,boxes,scores,delay,jmp)
% SHOWSORTEDBOXES(I,boxes,scores,delay,jmp)
if nargin < 3
    if size(boxes,2) < 5
        warning('no scores, showing boxes in given order');
        scores = 1:size(boxes,1);
    else
        warning('assuming scores in 5''th column of boxes');
        scores = boxes(:,5);
    end
elseif isempty(scores)
    scores = ones(1,size(boxes,1));
end


if nargin < 4
    delay = 0;
end
if nargin < 5
    jmp = 1;
end
n = size(boxes,1);
[r,ir] = sort(scores,'descend');
for t = 1:jmp:n
    clf; imagesc2(I);
    curBox = boxes(ir(t),:);
    plotBoxes(curBox);
    title(sprintf('%f (%d)',r(t),t));
    dpc(delay)
end
