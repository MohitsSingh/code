function showPtsOnImages(imgs,pts,pts_gt)


for p = 1:length(imgs)
    clf; imagesc2(imgs{p});
    if (length(size(pts))==3)
        plotPolygons(pts(:,:,p)','m.');
        plotPolygons(squeeze(pts_gt(p,:,1:2)),'g.');
    else
        plotPolygons(pts(:,p)','m.');
        plotPolygons(squeeze(pts_gt(p,1:2)),'g.');
    end
    drawnow
    %plotPolygons(factors(p)*xy_t(p,:),'g.');
    pause
end