function cameras = getCameraParams( world_to_cam_samples )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% load(P);
c1Params = cameraParameters('IntrinsicMatrix',[46300 0 0;0 46300 0;200 200 1]);
cameras = struct('cameraParams',{},'rot',{},'trans',{},'camMatrix',{});
%for t = 1:length(world_to_cam_samples);
for t = 1:length(world_to_cam_samples)
    imagePoints = world_to_cam_samples(t).cam;
    worldPoints = world_to_cam_samples(t).world;
    
%     xyz = world_to_cam_samples(t).world;
%     plot3(xyz(:,1),xyz(:,2),xyz(:,3),'r.');
    
    [rot, trans] =  extrinsics(imagePoints(1:15:end,:),...
        worldPoints(1:15:end,:), c1Params);
    
    cameras(t).cameraParams = c1Params;
    cameras(t).rot = rot;
    cameras(t).trans = trans;
    cameras(t).camMatrix = cameraMatrix(c1Params,rot,trans);
end

end

