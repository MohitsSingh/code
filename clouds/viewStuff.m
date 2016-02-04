% if ~exist('sufraces_in_timespace','var')
%     load surfaces.mat
% end
% S = sufraces_in_timespace; % can change this to surfaces_in_time
function viewStuff(S)

for t = 1:length(S)
    t
    xyz = S(t).xyz; % this is the actual computed surface
    
    
    x = xyz(:,1); y = xyz(:,2); z =xyz(:,3); % world coordinates
    clf;
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'.','LineWidth',1);
    hold on;
    uvw = S(t).uvw;
    if any(uvw)
        
        uvw_norms = sum(uvw.^2,2).^.5;
        %uvw_to_display = bsxfun(@rdivide,uvw,uvw_norms.^.5);
        uvw_to_display = uvw;
        u = uvw_to_display(:,1); v = uvw_to_display(:,2); w =uvw_to_display(:,3); % movement coordinates, only for t > 1
        quiver3(x,y,z,u,v,w,0,'g','LineWidth',2);
    end
    
    
    % fix the limits of the coordinate system
    xlim([0 5000]);
    ylim([0 5000]);
    zlim([0 5000]);
          az = 180;
        el = 90;
        view(az, el);
    grid on;
    %axis equal;
    
%     mesh = pointCloud2mesh(xyz);
    7
    %     az = 180;
    %     el = 0;
    %     view(az, el);
    pause
end