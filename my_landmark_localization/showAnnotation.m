function showAnnotation(phi,bb)
xy = squeeze(phi(:,:,1:2));
hold on;
plotPolygons(xy,'g.');

if (nargin >1)
    plotBoxes(bb);
end

if (size(phi,3)==3)
    isVisible =  squeeze(phi(:,:,3));
    plotPolygons(xy(isVisible==1,:),'r.');
end