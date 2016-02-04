function [res,T] = rectifyWindow(I,poly_source,target_rect_size)
% Transform a source quadrangular region in the source image A
% to a rectangle target_rect of size target_rect_size = [w h]

% crop image around the poly_source bounding box to make it faster.

% poly_box = pts2Box(poly_source);

r1 = target_rect_size(1);
r2 = target_rect_size(2);
target_rect = [0 0;r1 0;r1 r2;0 r2];
T = cp2tform(double(poly_source),double(target_rect),'affine');
% T = fitgeotrans(double(poly_source),double(target_rect),'affine');
% ra = imref2d(size2(I),[1 r1],[1 r2]);
% ra.XWorldLimits = [1 r1];
% ra.YWorldLimits = [1 r2];
% affine2d
% [x, y] = tformfwd(T, u, v);
res = imtransform(I,T,'bilinear','XData',[1 r1],'YData',[1 r2]);
%J = imtransform2(I,T.tdata.T,'show',1,'pad','replicate');
% J = imtransform2(I,[],'us',poly_source
%res = imwarp(I,T,'XData',[1 r1],'YData',[1 r2]);
% res = imwarp(I,ra,T);%,'XData',[1 r1],'YData',[1 r2]);


