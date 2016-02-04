function cam = makeCamera_t( t )
%MAKECAMERA_T Summary of this function goes here
%   Detailed explanation goes here
imageSize=[400 400];
% pixel size [micron] - 2.2
% fl_mm = 1.8;
cam_x = -220.8;
cam_y = [-1152, -1008, -864, -720, -576, -432, -288, -144, 0, 144, 288, 432, 576, 720, 864, 1008, 1152];
cam_z = 824;
zenith_angle = [124.957 ,128.459 ,132.568 ,137.379 ,142.956 ,149.255 ,155.944 ,161.965 ,164.775 ,162.171 ,...
    156.223 ,149.532,143.207 ,137.598 ,132.755 ,128.619 ,125.093];
azimuth_angle = [79.032, 77.516, 75.524, 72.798, 68.865, 62.773, 52.436, 33.300, 0.775, 327.796, 308.147, ...
    297.554, 291.338, 287.337, 284.572, 282.556, 281.02];
zenith_angle = 90-zenith_angle;

xyz_cam = [cam_x,cam_y(t),cam_z]*1000;
viewdir = pi*[azimuth_angle(t),zenith_angle(t),0]/180;
c = imageSize/2;
k = zeros(1,6);
p = zeros(1,2);
% D = A.LWC;
% D = D(110:310,110:310,:);
% f = find(D);
% f = f(1:end);
% xyz = ind2sub2(size(D),f);

% add to x,y,z the offset
world_center = [100 100 0];
xyz = bsxfun(@minus,xyz,world_center)*30; % meters

% look direction to xyz center....

xyz_center = mean(xyz);
newDir = xyz_center-xyz_cam;
% newDir = newDir/norm(newDir);

figure(1); clf
hold on;
plot3(xyz(:,1),xyz(:,2),xyz(:,3),'r.'); %plot3(xyz_cam(1),xyz_cam(2),xyz_cam(2),'g+');
% newDir = newDir*10^6;
plot3(xyz_cam(1),xyz_cam(2),xyz_cam(3),'g+');
quiver3(xyz_cam(1),xyz_cam(2),xyz_cam(3),newDir(1),newDir(2),newDir(3),0);
axis equal
newDir = newDir/norm(newDir);
[r1,r2,r3] = cart2sph(newDir(1),newDir(2),newDir(3));

[x1,y1,z1] = sph2cart(pi*[azimuth_angle(t)]/180.0,pi*zenith_angle(t)/180.0,1000000);
dd = 10^5;
quiver3(xyz_cam(1),xyz_cam(2),xyz_cam(3),dd*x1,dd*y1,dd*z1,'r')
newDir = [r1 -r2 0];

figure(2); hold on;
plot3(xyz(:,1),xyz(:,2),xyz(:,3),'r.'); plot3(xyz_cam(1),xyz_cam(2),xyz_cam(3),'g+');

end

