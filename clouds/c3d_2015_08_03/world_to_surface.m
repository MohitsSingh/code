function [ X,Y,Z,zgrid ] = world_to_surface( xyz,offset,f)
if nargin < 2
    offset = 131;
end
if nargin < 3
    f = 30;
end
dmap = xyz/f;
% z_ = 1;
rangeY=1:1:180;
rangeX=1:1:180;
[Y,X] = meshgrid(rangeX,rangeY);
Z = griddata(dmap(:,1),dmap(:,2),dmap(:,3),X,Y,'natural');

% [Z,X,Y] = gridfit(dmap(:,1),dmap(:,2),dmap(:,3),unique(X(:)),unique(Y(:)));

Z_ = zeros(480,480);
Z_(offset:size(Z,1)+offset-1,offset:size(Z,2)+offset-1) = Z;
Z = Z_;
Z = Z*30;
Z(Z<1860)=nan;
end

