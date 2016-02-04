function match_plot_x(I1Rect,I2Rect,inlierPoints1,inlierPoints2,toPermute,toSkip)
if nargin < 5
    toPermute = false;
end
if nargin < 6
    toSkip = 1;
end
if toPermute
    tt = randperm(size(inlierPoints1,1));
else
    tt = 1:size(inlierPoints1);
end
tt = tt(1:toSkip:length(tt));
for it = 1:size(inlierPoints1,1);
    
    t = tt(it);
    t
    clf;
    vl_tightsubplot(1,2,1);
    imagesc2(I1Rect); plotPolygons(inlierPoints1(t,:),'go');
    
    bb1 = inflatebbox([inlierPoints1(t,:) inlierPoints1(t,:)],40,'both',true);
    xlim_bb(bb1);
    
    vl_tightsubplot(1,2,2);
    imagesc2(I2Rect); plotPolygons(inlierPoints2(t,:),'go');
    bb2 = inflatebbox([inlierPoints2(t,:) inlierPoints2(t,:)],40,'both',true);
    xlim_bb(bb2);
    if it<size(inlierPoints1,1)
        dpc
    end
end
end