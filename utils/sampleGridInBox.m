function [x,y] = sampleGridInBox(bb,n,padding)
%SAMPLEGRIDINBOX Samples a meshgrid-style set of points inside a bounding
%box
if nargin < 3
    padding = 1;
end
x_range = linspace(bb(1)+padding,bb(3)-padding,n);
y_range = linspace(bb(2)+padding,bb(4)-padding,n);
[x,y] = meshgrid(x_range,...
    y_range);

if nargout == 1
    x = [x(:) y(:)];
end
    
end

