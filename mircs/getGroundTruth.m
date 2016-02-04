function [objectSamples,objectNames] = getGroundTruth(conf,train_ids,train_labels,exclusive)
% learn the different parts of the actions.
annotationPath = conf.annotationPath;
objectNames = {}; % list of all objects in this category.
objectSamples = {};
if (nargin < 4)
    exclusive = false;
end
n = 0;

objGT = struct('name',{},'bboxes',{});
if (isempty(train_labels))
    train_labels = true(size(train_ids));
end
for k = 1:length(train_ids)
%     k
    if (~train_labels(k))
        continue;
    end
    [~,name,~] = fileparts(train_ids{k});
    annotationFile = fullfile(annotationPath,[name '.xml']);
    if (~exist(annotationFile,'file'))
        continue;
    end
    L = loadXML(annotationFile);
    annotation = L.annotation;
    if (isfield(annotation,'object'))
        for iObject = 1:length(annotation.object)
            curObj = annotation.object(iObject);
            curPoly = poly2num(curObj.polygon);
            n = n+1;
            curObj.polygon = curPoly;
            curObj.sourceImage = train_ids{k};
            curObj.Orientation = 0;
            objFind = strncmp(objectNames,curObj.name,length(curObj.name));
            if (~any(objFind))
                objectNames = [objectNames,curObj.name];
                objFind = length(objectNames);
            else
                objFind = find(objFind);
            end
            curObj.partID = objFind;
            objectSamples{n} = curObj;
            %         conf.get_full_image = true;
            %         I = getImage(conf,'drinking_001.jpg');
            %         a.annotation.object
            %         LMplot(a.annotation,I*255)
        end
    end
end
end

% now train according to all parts independently.


function poly = poly2num(p) % convert from string representation of polygon to numerical.
poly.x = zeros(length(p.pt),1);
poly.y = zeros(length(p.pt),1);
for k = 1:length(p.pt)
    poly.x(k) = str2num(p.pt(k).x);
    poly.y(k) = str2num(p.pt(k).y);
end
end
