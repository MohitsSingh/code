function plotBoxes2(boxes,varargin)
%PLOTBOXES Summary of this function goes here
%   Detailed explanation goes here

tl = boxes(:,[2,1]);
br = boxes(:,[4,3]);
xxx = [];
yyy = [];
for k = 1:size(boxes,1)
    x0 = tl(k,1);
    y0 = tl(k,2);
    x1 = br(k,1);
    y1 = br(k,2);
    
    xx = [x0 x1 x1 x0 x0];
    yy = [y0 y0 y1 y1 y0];
    
    xxx = [xxx xx];
    yyy = [yyy yy];
    if (k < size(boxes,1))
        xxx = [xxx NaN];
        yyy = [yyy NaN];
    end
                    
end
plot(xxx,yyy,varargin{:});
end
