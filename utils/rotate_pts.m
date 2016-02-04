function pts = rotate_pts(pts,theta,centerPoint)
if (nargin < 3)
    centerPoint = [0 0];
end
curPts_c = bsxfun(@minus,pts,centerPoint);
%R = rotationMatrix(-pi*theta/180);
R = rotationMatrix(theta);
pts = bsxfun(@plus,curPts_c*R,centerPoint);