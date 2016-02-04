function myVisualize(conf,image_data,gt_mouth_corners,polys,poly_centers,angles)

for t = 1:length(image_data)
    I = getImage(conf,image_data(t));
    clf; imagesc2(I);
    plotPolygons(gt_mouth_corners{t},'r+');
    if (~isempty(polys))
        plotPolygons(polys(t),'r-');
    end
    if (~isempty(poly_centers))
        plotPolygons(poly_centers(t,:),'md','LineWidth',5);
    end
    if (~isempty(angles))
        quiver(poly_centers(t,1),poly_centers(t,2),10*sind(angles(t)),10*cosd(angles(t)));
        angles(t)
    end

    
    dpc;
end