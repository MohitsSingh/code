function pred = np_predict(I,I_rect,pts)
[xx,yy] = meshgrid(1:size(I,2),1:size(I,1));
bc = boxCenters(I_rect);
[h w a] = BoxSize(I_rect);
xx = (xx-bc(1))/w;
yy = (yy-bc(2))/h;

uv = [xx(:) yy(:)]';
pts = pts';
%D = l2(uv,pts);

forest = vl_kdtreebuild(pts);

[idx,dists] = vl_kdtreequery(forest,pts,uv,'numneighbors',10);
pred = zeros(size2(I));

pred(:)=sum(exp(-dists*30)/10,1);
% figure,imagesc2(pred)

%
%     R(:) = dists*w;
%     imagesc(R<.01);
%     %figure,imagesc2(exp(-R*10));

end