function [y_min,x_min] = fitCurveToImage(E,curve,xformParams)
A = (paintLines(zeros(size(E')),curve)>0)';
%     figure,imagesc(A)
%     figure,imagesc(E)

D = bwdist(E);
costFunction = -exp(-D/10);
%     figure,imagesc(costFunction);
[y,x] = find(A);
%     hold on;
%     plot(x,y,'r.');

[X,Y,Z] = meshgrid(xformParams.xRange,xformParams.yRange,xformParams.scaleRange);

mean_X = mean(x);
mean_Y = mean(y);

% mean_X = 0
%     mean_Y = 0
x_centered = x-mean_X;
y_centered = y -mean_Y;

grades = zeros(1,numel(X));
iMin = 1;
x_min = x;
y_min = y;
costMin = inf;
for k = 1:numel(X)
    
    offsetX  = X(k);
    offsetY = Y(k);
    scale = Z(k);
    
    x_ = x_centered*scale+mean_X+offsetX;
    y_ = y_centered*scale+mean_Y+offsetY;
    
    curInds = sub2ind(size(costFunction),round(y_),round(x_));
    grades(k) = sum(costFunction(curInds));
    if (grades(k) < costMin)
        costMin = grades(k);
        iMin = k;
        x_min = x_;
        y_min = y_;
%         disp(Z(k))  
%         clf;
%         imagesc(costFunction);
%         hold on;
%         plot(x_,y_,'r+');
%         plot(x_min,y_min,'g+');
%         title(num2str(costMin));
%         pause;
    end
    
end


clf;
imagesc(costFunction);
hold on;
%         plot(x_,y_,'r+');
plot(x_min,y_min,'g+');
title(num2str(costMin));

% pause;



end