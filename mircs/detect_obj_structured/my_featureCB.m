function psi = my_featureCB(param, x, y)
% extract a hog feature at a given point and rotation
% construct a window around the center and rotate it
cur_bb = inflatebbox(row(y([1 2 1 2])),param.windowSize,'both',true);

if (param.useRotations)
    box_pts = box2Pts(cur_bb);
    box_pts = rotate_pts(box_pts,pi*y(3)/180,y(1:2));    
    s = rectifyWindow(x.img,box_pts,param.windowSize);
%      clf; 
%      subplot(1,2,1);
%      imagesc2(x.img); plotPolygons(y(1:2),'r+','LineWidth',3,'MarkerSize',10); 
%      plotPolygons(box_pts);
%      subplot(1,2,2); imagesc2(s);
    drawnow;
else
% WARNING - for now, avoiding rotations for simplicity
    s = imResample(cropper(x.img, round(cur_bb)),param.windowSize,'bilinear');
end

% clf; subplot(1,2,1); imagesc2(x.img); plotPolygons(box_pts,'r-');    
% subplot(1,2,2); imagesc2(s);
mm = x.mouth_center/param.offset_factor;
yy = y(:,1:2)/param.offset_factor;
s2 = 1.4142135624;
mouth_offset = mm-yy;
mouth_offset = [mouth_offset.^2 s2*mouth_offset(:,1).*mouth_offset(:,2) s2*mouth_offset(:,1) s2*mouth_offset(:,2) ones(size(yy(:,1)))];
% mouth_offset = (y(1:2)-mm)/100;
%mouth_offset = [mouth_offset mouth_offset.^2 1];
% mouth_offset = [yy, yy.^2, -2*mm.*yy, mm.^2];
% [mouth_offset mouth_offset.^2 1];
if param.offset_ker_map
    mouth_offset = vl_homkermap(mouth_offset',1)';
end
if (~param.use_mouth_offset)
    mouth_offset = 0*mouth_offset;
end
psi = vl_hog(s,param.cellSize);
psi = [psi(:);mouth_offset(:)];
psi = sparse(double(psi));