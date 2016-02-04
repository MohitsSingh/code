function [ rects ] = makeTiles( I, nWindows, b)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    nWindows = 25;
end
if nargin < 3
    b = 2;
end
%wndSize = floor(size2(I)/sqrt(nWindows));
wndSize = floor(mean(size2(I))/sqrt(nWindows));
wndSize  = [wndSize wndSize ];
j = round(max(1,wndSize/b));
rects = {};
for dx = 1:j(2):size(I,2)-wndSize(2)+1
    for dy = 1:j(1):size(I,1)-wndSize(1)+1
        rects{end+1} = [dx dy dx+wndSize(2)-1 dy+wndSize(1)-1];
    end    
end
rects = cat(1,rects{:});
rects(:,3:4) = rects(:,3:4)+1;
rects = BoxIntersection(rects,[1 1 fliplr(size2(I))]);