function int_box = getInteractionRegion(prevBox,curBox)
% int_box = BoxIntersection(prevBox,curBox);
% if int_box(1)==-1
% if boxes intersect, all is fine.
% otherwise, find where the line crosses
%int_point = mean(boxCenters([prevBox;curBox]));
center1 = boxCenters(prevBox);
center2 = boxCenters(curBox);
r1 = (prevBox(3)-prevBox(1))/2;
r2 = (curBox(3)-curBox(1))/2;
v = center2-center1;
d = norm(v)-r1-r2;
int_box = center1+(r1+d/2)*v/norm(v);
% end
[~,~,a1] = BoxSize(prevBox);
[~,~,a2] = BoxSize(curBox);
curScale = (a1.^.5+a2.^.5)/4; % half of mean
int_box = round(inflatebbox(int_box,curScale,'both',true));