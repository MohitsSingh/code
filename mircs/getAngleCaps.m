function cp = getAngleCaps(sz,thetaRange)
[xx,yy] = meshgrid(1:sz(1),1:sz(2));
cy = 0;
cx = 25;
maxRad = 15;
theta = 180-180*atan2(yy-cy,xx-cx)/pi;
radCap = ((yy-cy).^2+(xx-cx).^2).^.5 <= maxRad;
dTheta = 10;
wTheta = dTheta/2;
minTheta = 40;
% thetaRange = 0:10:90;
cp = zeros([sz length(thetaRange)]);
for qq = 1:length(thetaRange)
    if (thetaRange(qq) < (180-minTheta) && thetaRange(qq) > minTheta)
        s = (theta <= thetaRange(qq)+wTheta) & (theta>=thetaRange(qq)-wTheta);
        cp(:,:,qq) = imdilate(s,ones(1,9)).*radCap;
    end
end
