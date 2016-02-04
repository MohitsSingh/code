function bb = getSingleRect(my_bbox_format)
if (nargin ==0)
    my_bbox_format = false;
end
% grabs a single rectangle from the user
[~,api]=imRectRot('rotate',0);
bb = api.getPos();
if (my_bbox_format)
    bb(3:4) = bb(3:4)+bb(1:2);
end
bb = bb(1:4);