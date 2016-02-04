function XY = project_to_image(xyz,cam)

XY = [xyz ones(size(xyz,1),1)]*cam.camMatrix;
XY = XY(:,1:2)./XY(:,[3 3]);


end

