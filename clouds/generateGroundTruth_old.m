function world_to_cam_samples = generateGroundTruth(A,offset,z_offset)
if nargin == 1
    offset = 0;
end
if nargin < 3
    z_offset = 0;
end
imageSize=[400 400];
cam_x = -220.8;
cam_y = [-1152, -1008, -864, -720, -576, -432, -288, -144, 0, 144, 288, 432, 576, 720, 864, 1008, 1152];
cam_z = 824;
zenith_angle = [124.957 ,128.459 ,132.568 ,137.379 ,142.956 ,149.255 ,155.944 ,161.965 ,164.775 ,162.171 ,...
    156.223 ,149.532,143.207 ,137.598 ,132.755 ,128.619 ,125.093]*pi/180;
azimuth_angle = [79.032, 77.516, 75.524, 72.798, 68.865, 62.773, 52.436, 33.300, 0.775, 327.796, 308.147, ...
    297.554, 291.338, 287.337, 284.572, 282.556, 281.02]*pi/180;
zenith_angle = pi/2-zenith_angle;
makeCalibrationPattern = false;
useOnlyVisiblePoints = true;
%%
    
    D = A.LWC;
    %D = D(130:329,130:329,1:185);
    
    D = D([130:329]+offset,[130:329]+offset,50:185);
    D_orig = D;
    % find only the top-level z value
    onlySurface = true;
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
        N = size(D);
        [x,y,z] = meshgrid(1:d1:N(1),1:d1:N(2),1:d1:N(3));
%         [x,y,z] = meshgrid(1:N:size(D,1),1:N:size(D,2),1);        
        xyz = [x(:) y(:) z(:)];
        
    end
    % add to x,y,z the offset
    world_center = [101 101 1];
    ddd = 1:1:size(xyz,1);
    world_center = world_center*30;
    xyz(:,3) = xyz(:,3)-z_offset;
    xyz_orig(:,3) = xyz_orig(:,3)-z_offset;
    xyz(:,3) = xyz(:,3)+49;
    xyz_orig(:,3) = xyz_orig(:,3)+49;
    xyz = xyz*30;    
    xyz_orig = xyz_orig*30;
    
    %%
%     close all
    world_to_cam_samples = struct('t',{},'world',{},'cam',{},'projMat',{},'depth_gt',{});
    all_in_frame = true(size(xyz,1),1);
    for t =1:length(cam_y)
        xyz_cam = [cam_x,cam_y(t),cam_z]*1000.0;
        viewdir = [azimuth_angle(t),-zenith_angle(t),0];
        c = imageSize/2;
        k = zeros(1,6);
        p = zeros(1,2);
        R = norm(xyz_cam-world_center);
        [x1,y1,z1] = sph2cart(azimuth_angle(t),zenith_angle(t),R);
        dd = 10^5;
        %     quiver3(xyz_cam(1),xyz_cam(2),xyz_cam(3),x1,y1,z1,'r');
        camera_f = 46300*[1 1];
        cam = camera(xyz_cam,imageSize,viewdir,camera_f,c,k,p);
        % build a projection matrix which does the same
        
        
        %[uv_d,depth_d,inframe_d,projMat_d]=cam.project(xyz_orig);
        %
        
        if useOnlyVisiblePoints
            % find visibility of points
            xyz = xyz_orig;
            visiblePtInds=HPR(xyz,xyz_cam,5);
            xyz = xyz(visiblePtInds,:);            
            [uv,depth,inframe,projMat]=cam.project(xyz);
        else
            [uv,depth,inframe,projMat]=cam.project(xyz);
        end
        
        FF = 1;
        f_in_frame = row(find(inframe));
        %     f_in_frame = vl_colsubset(f_in_frame,2000,'Uniform');
        %f_in_frame = f_in_frame(1:400:end);
        xyz_in = xyz(f_in_frame,:);
        uv_in = uv(f_in_frame,:);
        z = accumarray(round(FF*uv(inframe,[2 1])),1,FF*imageSize);
        z = imResample(z,1/FF);
        
%         plot3(xyz(visiblePtInds,1),xyz(visiblePtInds,2),xyz(visiblePtInds,3),'r.')
%         uv1 = round(uv);%         
%         % compute distance to the camera for "cheap" visibility
%         dists_to_cam = l2(xyz_in,xyz_cam);        
%         img_inds = sub2ind2(size2(z),uv1);
%         zz = zeros(size2(z));        
%         zz(img_inds) = dists_to_cam;
%         best_ind_map = zeros(size2(z));
%         for ii = 1:size(uv,1)            
%             cur_location = img_inds(ii);
%             cur_dist = dists_to_cam(ii);            
%             if (zz(cur_location) > cur_dist)
%                 zz(cur_location) = cur_dist;
%                 best_ind_map(cur_location) = ii;
%             end            
%         end    
%                 
%         [i,j,v] = find(best_ind_map);
%         
%         cam.optimizecam(xyz_in(v,:),        
        world_to_cam_samples(t).t = t;
        world_to_cam_samples(t).world = xyz_in;
        world_to_cam_samples(t).cam = uv_in;
        world_to_cam_samples(t).projMat = projMat;
%         clf; imagesc2(z);
%         pause;
%         close all;
        %%x2(z.^.5);
%         z1 = z>0;
%         z1 = normalise(z1);
%         
%         z2 = im2single(imread('Images_divided_by_maxValue/Image_T_01_A_01.png'));
%         z2 = z2 > .1;
%         clf; imagesc2(z1+z2);
%         
%         break
        %clf;
        
        %     figure(1); clf;
        %     z = z/max(z(:));
        %     %     subplot(1,2,1);
        %     imagesc2(z);
        %         dpc(.1)
    end
    %     x2(z)
    
end
