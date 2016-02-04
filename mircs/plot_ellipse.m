function plot_ellipse(ellipse_t)
% rotation matrix to rotate the axes with respect to an angle phi
    cos_phi = cos(ellipse_t.phi);
    sin_phi = sin(ellipse_t.phi);
    ellipse_t.R = [ cos_phi sin_phi; -sin_phi cos_phi ];
    
    % the axes
    ver_line        = [ [ellipse_t.X0 ellipse_t.X0]; ellipse_t.Y0+ellipse_t.b*[-1 1] ];
    horz_line       = [ ellipse_t.X0+ellipse_t.a*[-1 1]; [ellipse_t.Y0 ellipse_t.Y0] ];
    new_ver_line    = ellipse_t.R*ver_line;
    new_horz_line   = ellipse_t.R*horz_line;
    
    % the ellipse
    theta_r         = linspace(0,2*pi);
    ellipse_x_r     = ellipse_t.X0 + ellipse_t.a*cos( theta_r );
    ellipse_y_r     = ellipse_t.Y0 + ellipse_t.b*sin( theta_r );
    rotated_ellipse = ellipse_t.R * [ellipse_x_r;ellipse_y_r];
    
    % draw
%     hold_state = get( axis_handle,'NextPlot' );
%     set( axis_handle,'NextPlot','add' );
    plot( new_ver_line(1,:),new_ver_line(2,:),'r' );
    plot( new_horz_line(1,:),new_horz_line(2,:),'r' );
    plot( rotated_ellipse(1,:),rotated_ellipse(2,:),'r' );
end