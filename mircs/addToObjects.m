function imgData = addToObject(imgData,objName,xy)
objects = imgData.objects;
bads = false(size(objects));
for  tt = 1:length(objects)
    if isempty(objects(tt).poly) && isempty(objects(tt).name)
        bads(tt) = true;
    end
end
objects = objects(~bads);
p = length(objects);
objects(p+1).name = objName;
objects(p+1).poly = xy;
imgData.objects = objects;