function res = inBox(rect,x,y)
if (size(rect,2)==2) % rect given in w h form
    rect = [1 1 rect];
elseif numel(rect) > 4
    rect = [1 1 fliplr(size2(rect))];
end
if(nargin<3)
    y = x(:,2);
    x = x(:,1);
end
res = x >= rect(1) & y >=rect(2) & x <= rect(3) & y <= rect(4);
end