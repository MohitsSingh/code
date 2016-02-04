function [r,ovp] = fitRectangle(roiMask,lite)
% FITRECTANGLE find best fit of an oriented rectangle to region given by
% mask.
if nargin < 2
    lite = true;
end
% initialize with the ellipse of the polygon
rr = regionprops(roiMask,'MajorAxisLength','MinorAxisLength','Orientation','Area','Centroid');
height0 = rr.MajorAxisLength;
width0 = rr.MinorAxisLength;
%theta0 = pi/2+pi*rr.Orientation/180;
theta0 = rr.Orientation;
centerPoint = rr.Centroid;
u = [cosd(theta0),-sind(theta0)];
p0 = centerPoint-height0*u/2;
p1 = centerPoint+height0*u/2;
roiPoly = fliplr(bwtraceboundary2(roiMask));
roiPoly = poly2cw2(roiPoly);
roiPoly = simplifyPolygon(roiPoly,.5);
roiArea = polyarea2(roiPoly);
x0 = [p0,p1,width0];
if (lite)
    r = x0;
    
    dist_to_center = l2([x0(1:2);x0(3:4)],fliplr(size2(roiMask)/2));
    if dist_to_center(2)<dist_to_center(1)
        r = x0([3 4 1 2 5]);
    end
    
    r = directionalROI_rect_lite(x0);
    ovp = polyOverlap(r,roiPoly);
    r = x0;
    return;
end
options = optimset('fminsearch');
options.TolFun = .01;
options.TolX = .01;
[r,fval,exitflag,output] = fminsearch(@(x) polyRectOvp(x,roiPoly),...
    x0,options);
% r = directionalROI_rect_lite(r);
ovp = 1-fval;

%        f = @(x,c) x(1).^2+c.*x(2).^2;  % The parameterized function.
%        c = 1.5;                        % The parameter.

    function f = polyRectOvp(x,roiPoly)
        xy_rect = directionalROI_rect_lite(x);
        curArea = polyarea2(xy_rect);
        curInt = polybool2('&', xy_rect,roiPoly);
        intArea = polyarea2(curInt);
        f = 1-(intArea/(curArea+roiArea-intArea));
        %         f = 1-(intArea/(curArea));
        %                 clf; imagesc2(roiMask);
        %                 plotPolygons(xy_rect);
        %                 title(num2str(f));
        %                 drawnow
        %                 dpc(.01);
    end
end