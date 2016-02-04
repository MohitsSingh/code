% IMAGEPT2PLANE - Project image points to a plane and return their 3D locations
%
% Usage:  pt = imagept2plane(C, xy, planeP, planeN, ik1, ik2)
%
% Arguments:  
%          C - Camera structure, see CAMSTRUCT for definition.  Alternatively
%              C can be a 3x4 camera projection matrix.
%         xy - Image points specified as 2 x N array (x,y) / (col,row)
%     planeP - Some point on the plane.
%     planeN - Plane normal vector.
%   ik1, ik2 - Optional inverse lens distortion parameters obtained via
%              FINDINVERSELENSDIST 
%
% Returns:
%         pt - 3xN array of 3D points on the plane that the image points
%              correspond to. 
%
% Note that the plane is specified in terms of the world frame by defining a
% 3D point on the plane, planeP, and a surface normal, planeN.
% 
% See also CAMSTRUCT, CAMERAPROJECT, FINDINVERSELENSDIST

% Copyright (c) 2015 Peter Kovesi
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

% PK May 2015

function pt = imagept2plane(C, xy, planeP, planeN, ik1, ik2)
    
    [dim,N] = size(xy);
    if dim ~= 2
        error('Image xy data must be a 2 x N array');
    end
    
    if ~isstruct(C) && all(size(C) == [3,4])  % We have a projection matrix
        C = projmatrix2camstruct(C);          % Convert to camera structure
    end
        
    % Reverse the projection process as used by CAMERAPROJECT

    % If only one focal length specified in structure use it for both fx and fy
    if isfield(C, 'f')
        fx = C.f;
        fy = C.f;
    elseif isfield(C, 'fx') && isfield(C, 'fy')
        fx = C.fx;
        fy = C.fy;        
    else
        error('Invalid focal length specification in camera structure');
    end    

    if isfield(C, 'skew')     % Handle optional skew specfication
        skew = C.skew;
    else
        skew = 0;
    end    
    
    % Subtract principal point and divide by focal length to get normalised,
    % distorted image coordinates.  Note skew represents the 2D shearing
    % coefficient times fx
    y_d = (xy(2,:) - C.ppy)/fy;   
    x_d = (xy(1,:) - C.ppx - y_d*skew)/fx;   

    % Apply inverse lens distortion parameters if they have been supplied.
    if exist('ik1', 'var') && exist('ik2', 'var')
        r_dsqrd = x_d.^2 + y_d.^2;
        D = 1 + ik1*r_dsqrd + ik2*r_dsqrd.^2;
        x_n = D.*x_d;
        y_n = D.*y_d;
    else
        x_n = x_d;   
        y_n = y_d;
    end
    
    ray = [x_n; y_n; ones(1,N)];  % These are the points at z = 1 from the
                                  % principal point, the viewing rays
                                  % described in terms of the camera frame.
                          
    % Rotate to get the viewing rays in the world frame
    ray = C.Rc_w'*ray;      
    
    % The point of intersection of each ray with the plane will be 
    %   pt = C.P + k*ray    where k is to be determined
    %
    % Noting that the vector planeP -> pt will be perpendicular to planeN.
    % Hence the dot product between these two vectors will be 0.  
    %  dot((( C.P + k*ray ) - planeP) , planeN)  = 0
    % Rearranging this equation allows k to be solved
    
    pt = zeros(3,N);
    for n = 1:N
        k = (dot(planeP, planeN) - dot(C.P, planeN)) / dot(ray(:,n), planeN);
        pt(:,n) = C.P + k*ray(:,n);
    end