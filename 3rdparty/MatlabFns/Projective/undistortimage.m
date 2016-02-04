% UNDISTORTIMAGE - Removes lens distortion from an image
%
% Usage:  nim = undistortimage(im, f, ppx, ppy, k1, k2)
%
% Arguments: 
%           im - Image to be corrected.
%            f - Focal length in terms of pixel units 
%                (focal_length_mm/pixel_size_mm)
%     ppx, ppy - Principal point location in pixels.
%       k1, k2 - Radial lens distortion parameters.
%
% Returns:  
%          nim - Corrected image.
%
%  It is assumed that k1, k2 are computed/defined with respect to normalised
%  image coordinates corresponding to an image plane 1 unit from the projection
%  centre.

% Copyright (c) 2010 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% peter.kovesi at uwa edu au
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% October  2010  Original version
% November 2010  Bilinear interpolation + corrections
% April    2015  Cleaned up and speeded up via use of interp2

function nim = undistortimage(im, f, ppx, ppy, k1, k2)
    
    % Strategy: Generate a grid of coordinate values corresponding to an ideal
    % undistorted image.  We then apply the imaging process to these
    % coordinates, including lens distortion, to obtain the actual distorted
    % image locations.  In this process these distorted image coordinates end up
    % being stored in a matrix that is indexed via the original ideal,
    % undistorted coords.  Thus for every undistorted pixel location we can
    % determine the location in the distorted image that we should map the grey
    % value from.

    % Start off generating a grid of ideal values in the undistorted image.
    [rows,cols,chan] = size(im);        
    [xu,yu] = meshgrid(1:cols, 1:rows);
    
    % Convert grid values to normalised values with the origin at the principal
    % point.  Dividing pixel coordinates by the focal length (defined in pixels)
    % gives us normalised coords corresponding to z = 1
    x = (xu-ppx)/f;
    y = (yu-ppy)/f;    

    % Apply lens distortion
    r2 = x.^2+y.^2;                % Squared normalized radius
    d = 1.0 + k1.*r2 + k2.*r2.^2;  % Distortion scaling factor
    x = x.*d;
    y = y.*d;
    
    % Now rescale by f and add the principal point back to get distorted x
    % and y coordinates
    xd = x*f + ppx;
    yd = y*f + ppy;
    
    % Interpolate values from distorted image to their ideal locations
    if ndims(im) == 2   % Greyscale
        nim = interp2(xu,yu,double(im),xd,yd); 
    else % Colour
        nim = zeros(size(im));
        for n = 1:chan
            nim(:,:,n) = interp2(xu,yu,double(im(:,:,n)),xd,yd); 
        end
    end

    if isa(im, 'uint8')      % Cast back to uint8 if needed
        nim = uint8(nim);
    end
    
