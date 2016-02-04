function [y_min,x_min] = fitCurveToImage(E,curve,xformParams)
A = (paintLines(zeros(size(E')),curve)>0)';
%     figure,imagesc(A)
%     figure,imagesc(E)

D = bwdist(E);
costFunction = -exp(-D);
%     figure,imagesc(costFunction);
[y,x] = find(A);
%     hold on;
%     plot(x,y,'r.');

%[X,Y,Z] = meshgrid(xformParams.xRange,xformParams.yRange,xformParams.scaleRange);

mean_X = mean(x);
mean_Y = mean(y);

x_centered = x-mean_X;
y_centered = y -mean_Y;

x0 = [0 0 1 1];
% % options = [];
% % options.Display = 'iter';
options = optimset;
options.MaxIter = 1000;
[x_min,fval] = fminsearch(@curveFit,x0);
 [x_,y_] = makeCurve(x_min);
%
%     if (grades(k) < costMin)
%         costMin = grades(k);
%         iMin = k;
%         x_min = x_;
%         y_min = y_;
%         disp(Z(k))
        clf;
        imagesc(costFunction);
        hold on;
        plot(x_,y_,'r+');
%         plot(x_min,y_min,'g+');
        title(num2str(fval));
        pause;
%     end


    function cost = curveFit(x)
        [x_,y_] = makeCurve(x);
        
        curInds = sub2ind(size(costFunction),round(y_),round(x_));
        cost= sum(costFunction(curInds));
        
        
        
    end

    function [x_,y_] = makeCurve(x)
        offsetX  = x(1);
        offsetY = x(2);
        scaleX = x(3);
        scaleY = x(4);
        
        x_ = x_centered*scaleX+mean_X+offsetX;
        y_ = y_centered*scaleY+mean_Y+offsetY;
    end

end