function bads = findWeirdRoutes(routes,boxes)
xy = boxCenters(boxes);
nRoutes = size(routes,1);
prevDirection = zeros(nRoutes,2);
%prevPoints = xy(routes(:,1),:);
goods = true(nRoutes,1);
for t = 2:size(routes,2)
    curDirection = xy(routes(:,t),:)-xy(routes(:,t-1),:);
    goods = goods & sum(curDirection.*prevDirection,2) >= 0;
    prevDirection = curDirection;
end

bads = ~goods; % :-)