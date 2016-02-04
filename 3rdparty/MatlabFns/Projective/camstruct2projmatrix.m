% CAMSTRUCT2PROJMATRIX
%
% Usage: P = camstruct2projmatrix(C)
%
% Argument: C - Camera structure.
% Returns:  P - 3x4 camera projection matrix that maps homogeneous 3D world 
%               coordinates to homogeneous image coordinates.
%
% Function takes a camera structure and returns its equivalent projection matrix
% ignoring lens distortion parameters etc
%
% See also: CAMSTRUCT, PROJMATRIX2CAMSTRUCT, CAMERAPROJECT

% Copyright (c) Peter Kovesi
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

% PK April 2015

function P = camstruct2projmatrix(C)
    
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
    
    if isfield(C, 'skew')  % Handle optional skew specification
        skew = C.skew;
    else
        skew = 0;
    end
    
    K = [fx  skew   C.ppx  0
         0    fy    C.ppy  0
         0    0       1    0];
    
    T = [ C.Rc_w  -C.Rc_w*C.P
         0  0  0       1     ];
    
    P = K*T;