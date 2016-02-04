function z = getLogPolarMask(radius,nTheta,nLayers)

if (nargin < 3)
    nLayers = 3;
end


if isscalar(radius)
    r = round(radius*.3*(2.^(0:nLayers)));
else
    r = radius;
end
if (nargin < 2)
    nTheta = 10;
end

dTheta = pi/(nTheta/2);
theta = 0:dTheta:2*pi;
z = [];
for iTheta = 1:length(theta)-1
    if (iTheta == 1)
        z = maskCircle(theta(iTheta),dTheta,r(end));
    else
        z = max(z,iTheta*maskCircle(theta(iTheta),dTheta,r(end)));
    end
end
z_dist = zeros(size(z));
z_dist(round(size(z,1)/2),round(size(z,2)/2)) = 1;
z_dist = bwdist(z_dist);
zz = z > 0;
for rr = r(end-1:-1:2)
    z(zz & z_dist <= rr) = z(zz & z_dist <= rr) + length(theta)-1;
end
z(z_dist <= r(1)) = max(z(:)+1);%max(z(:)) + 2-length(theta);
end