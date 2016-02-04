function [ rotated_ellipse ] = ellipse_points( ellipse_t )
%ELLIPSE_POINTS Summary of this function goes here
%   Detailed explanation goes here
orientation_rad = ellipse_t.phi;
cos_phi = cos( orientation_rad );
sin_phi = sin( orientation_rad );
R = [ cos_phi sin_phi; -sin_phi cos_phi ];

X0 = ellipse_t.X0;
Y0 = ellipse_t.Y0;
b = ellipse_t.b;
a = ellipse_t.a;


% the axes
ver_line        = [ [X0 X0]; Y0+b*[-1 1] ];
horz_line       = [ X0+a*[-1 1]; [Y0 Y0] ];
new_ver_line    = R*ver_line;
new_horz_line   = R*horz_line;

% the ellipse
theta_r         = linspace(0,2*pi);
ellipse_x_r     = X0 + a*cos( theta_r );
ellipse_y_r     = Y0 + b*sin( theta_r );
rotated_ellipse = R * [ellipse_x_r;ellipse_y_r];

end

