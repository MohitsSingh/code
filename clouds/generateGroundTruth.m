function [world_to_cam_samples,patterns] = generateGroundTruth(A,offset,z_offset,R)
if nargin == 1
    offset = 0;
end
if nargin < 3
    z_offset = 0;
end
patterns = [];
imageSize= [400 400];
cam_x = -220.8;
%cam_y = [-1152, -1008, -864, -720, -576, -432, -288, -144, 0, 144, 288, 432, 576, 720, 864, 1008, 1152];
%cam_y = [-1152, -1008, -864, -720, -576, -432, -288, -144, 0, 144, 288, 432, 576, 720, 864, 1008, 1152];
cam_y = -2304:144:2304;
cam_z = 824;
zenith_angle = [124.957 ,128.459 ,132.568 ,137.379 ,142.956 ,149.255 ,155.944 ,161.965 ,164.775 ,162.171 ,...
    156.223 ,149.532,143.207 ,137.598 ,132.755 ,128.619 ,125.093]*pi/180;
% azimuth_angle = [79.032, 77.516, 75.524, 72.798, 68.865, 62.773, 52.436, 33.300, 0.775, 327.796, 308.147, ...
%     297.554, 291.338, 287.337, 284.572, 282.556, 281.02]*pi/180;
% zenith_angle = pi/2-zenith_angle;
makeCalibrationPattern = false;
camera_f = 46300*[1 1]/1.0075;
useOnlyVisiblePoints = true;
load('Aviad_Calib/ground_points_and_values.mat');
load('Aviad_Calib/Calib_images.mat'); % Img
% B = load('ConvectiveCloud_Images');
%%
D = A.LWC;
%D = D(130:329,130:329,1:185);

D = D([130:329]+offset,[130:329]+offset,50:185);% was 50:185
D_orig = D;
% find only the top-level z value
onlySurface = false;
if onlySurface
    D_ = zeros(size(D));
    [ii,jj] = find(max(D,[],3));
    
    for kk = 1:length(ii)
        i_ = ii(kk);
        j_ = jj(kk);
        curCol = squeeze(D(i_,j_,:));
        [fi,fj,v] = find(curCol,1,'last');
        D_(i_,j_,fi) = v;
        % end
    end
    D = D_;
end

% find the surface elevation.
[X,Y] = meshgrid(1:size(D,1),1:size(D,2));
f = find(D);
xyz = ind2sub2(size(D),f);
xyz_orig = ind2sub2(size(D_orig),find(D_orig));
% xyz = xyz-10;
% make a calibration pattern
if makeCalibrationPattern
    d1 = 10;
    %         N = size(D);
    %         [x,y,z] = meshgrid(1:d1:N(1),1:d1:N(2),1:d1:N(3));
    [x,y,z] = meshgrid(1:d1:size(D,1),1:d1:size(D,2),50);
    xyz = [x(:) y(:) z(:)];
end
% add to x,y,z the offset

%world_center = [101 101 50];
world_center = [([101 101]+z_offset), 50+1]
world_center = world_center*30;
xyz(:,3) = xyz(:,3);%-z_offset;
xyz_orig(:,3) = xyz_orig(:,3);%-z_offset;
xyz(:,3) = xyz(:,3)+49;
xyz_orig(:,3) = xyz_orig(:,3)+49;
xyz = xyz*30;
xyz_orig = xyz_orig*30;
% % %     load('calibration_points');
% % %     xyz = calibration_points;
% % %     xyz(:,3) = xyz(:,3)+49;
% % %     xyz = calibration_points*30;xyz_orig = xyz;
%     p_top = [66   109    63];
%     k = convhull(xyz(:,1),xyz(:,2),xyz(:,3));
%     xyz = xyz(k,:);
%     plot3(xyz(:,1),xyz(:,2),xyz(:,3),'r.')
%%
world_to_cam_samples = struct('t',{},'world',{},'cam',{},'projMat',{},'depth_gt',{});
for t =1:length(cam_y)
    t
    % for t = 5
    xyz_cam = [cam_x,cam_y(t),cam_z]*1000.0;
    c = imageSize/2;
    k = zeros(1,6);
    p = zeros(1,2);
    view_dir = world_center-xyz_cam;
    view_dir = view_dir/norm(view_dir);
    
        
    [az,elev,~] = cart2sph(view_dir(1),view_dir(2),view_dir(3));
    viewdir = [az -elev 0];
    
    %     viewdir = [azimuth_angle(t),-zenith_angle(t),0];
    
    %         warning('was -elev')
    
    cam = camera(xyz_cam,imageSize,viewdir,camera_f,c,k,p);
    
    if useOnlyVisiblePoints
        % find visibility of points
        xyz = xyz_orig;
        visiblePtInds=HPR(xyz,xyz_cam,5); % 5
        xyz = xyz(visiblePtInds,:);
        [uv,depth,inframe,projMat]=cam.project(xyz);
    else
        [uv,depth,inframe,projMat]=cam.project(xyz);
    end
    
    %
    %plotPolygons(p_,'g.');%
    
    FF = 1;
    f_in_frame = row(find(inframe));
    xyz_in = xyz(f_in_frame,:);
    uv_in = uv(f_in_frame,:);
    %p_target = [213 161];
    %xyz_target = [3510        3360        2670];
    %         p_target = [134 199];
    %         xyz_target = [870        2550        2160];
    p_target = [165 153];
    xyz_target = [ 1950        3240        3270];
    diffs_to_target = vec_norms(bsxfun(@minus,uv_in,p_target));
    [d,id] = min(diffs_to_target);
    %         xyz_in(id,:)
    
    p_target1 = cam.project(xyz_target);
    bb = inflatebbox([p_target1 p_target1],[20 20],'both',true);
%     z2 = im2single(imread(sprintf('Images_divided_by_maxValue/Image_T_01_A_%02.0f.png',t)));
    %%
    % % %
    % % %
    % % %     %             tt
    % % %     ppp = Points;
    % % %     p_test = cam.project(ppp(:,[1 2 3]));
    % % %     II = im2double(Img{t});
    % % %     II = imrotate(II,-90);
    % % %     II(II<0) = 0;
    % % %     clf; imagesc2(II.^.25);
    % % %     plotPolygons(p_test,'r.');
    % % %     %
    % % %
    % % %     dpc;continue
    %%
    % % %
    % % %
    % % %             for iz = 1:5
    % % %
    % % %                 p_test = cam.project(ppp(iz,[1 2 3]));
    % % %                 %             p_test = fliplr(p_test);
    % % %                 %
    % % %
    % % %
    % % %                 II = normalise(II);
    % % %
    % % %
    % % %                 dpc;
    % % %             end
    % % %         end
    %
    
    %         dpc;
    % % %
    % %     clf;imagesc2(z2);
    % %     plotPolygons(p_target1,'g.');
    % % %     plotPolygons(p_test,'r.');
    % %     %          plotPolygons(uv_in(1:10:end,:),'m.');
    % %     dpc;
    % %         drawnow
    %         p_target2 = cam.project(world_center)
    
    %         xlim_bb(bb);
    z = accumarray(round(FF*uv(inframe,[2 1])),1,FF*imageSize);
    z = imResample(z,1/FF);
    
    
    %     clf;
    %     subplot(1,2,1);
    %     imagesc2(z);
    %     subplot(1,2,2);
    %     imagesc2(imrotate(B.Img{1,t},-90));
    %     dpc;
    %
    %         patterns{t} = z;
    %
    %         figure(1);clf; imagesc2(z);
    %         figure(2);clf; imagesc2(z2);
    %         dpc;continue;
    
    world_to_cam_samples(t).t = t;
    world_to_cam_samples(t).world = xyz_in;
    world_to_cam_samples(t).cam = uv_in;
    world_to_cam_samples(t).camStruct = cam;
    world_to_cam_samples(t).projMat = projMat;
    world_to_cam_samples(t).xyz_cam = xyz_cam;
    world_to_cam_samples(t).world_center = world_center;
    world_to_cam_samples(t).p_target1 = p_target1;
    world_to_cam_samples(t).xyz_target1 = xyz_target;
    
end


end
