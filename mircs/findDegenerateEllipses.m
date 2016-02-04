function [d] = findDegenerateEllipses(ellipses,tol)
d = false(size(ellipses,1),1);
if (nargin < 2)
    tol = 1;
end
for iEllipse = 1:size(ellipses,1)
    a = ellipses(iEllipse,:);
    [~,x,y] = plotEllipse2(a(1),a(2),a(3),a(4),a(5:7),'g',100,2,[],false);    
%     [xx,yy] = LineTwoPnts(x(1),y(1),x(end),y(end));
%     
%     hold on; plot(xx,yy,'r');
    
    % find distance between all point of ellipse and straight line between
    % it's edges. 
    x1 = x(1); y1 = y(1); x2 = x(end); y2 = y(end);
    seg = createEdge(x1,y1,x2-x1,y2-y1);
%     figure,plot(x,y);hold on;
%     drawEdge(seg,'color','g')
    [dis] = distancePointEdge([x(:) y(:)],seg);
    
    if (max(dis) <= tol)
        d(iEllipse) = true;
    end            
%     proj_xy = projPointOnLine([x(:) y(:)],createLine(x(1),y(1),x(end)-x(1),y(end)-y(1)));           
end