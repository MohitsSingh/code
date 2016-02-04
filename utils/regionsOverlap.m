function [ovp,ints,uns] = regionsOverlap(regions1,regions2,usebb)
if (nargin == 1)
    regions2 = regions1;
end
if nargin < 3
    usebb = true;
end


if (~iscell(regions1)), regions1 = {regions1}; end
if (~iscell(regions2)), regions2 = {regions2}; end


% to make it more efficient, find the bounding box containing all regions


ovp = zeros(length(regions1),length(regions2));
ints = zeros(length(regions1),length(regions2));
uns = zeros(length(regions1),length(regions2));

% get the bounding boxes first.
if islogical(usebb) % then it must be a box
    boxes1 = getRegionBoxes(regions1);
    boxes2 = getRegionBoxes(regions2);
    if usebb        
        new_bb = BoxUnion([boxes1;BoxUnion(boxes2)]);
        regions1 = cellfun2(@(x) cropper(x,new_bb),regions1);
        regions2 = cellfun2(@(x) cropper(x,new_bb),regions2);
        boxes1 = BoxIntersection(boxes1,new_bb);
        boxes2 = BoxIntersection(boxes2,new_bb);
    end
else % given a box :-)
    new_bb = usebb;
    regions1 = cellfun2(@(x) cropper(x,new_bb,true),regions1);
    regions2 = cellfun2(@(x) cropper(x,new_bb,true),regions2);
end

%
boxOverlaps = boxesOverlap(boxes1,boxes2);

for i1 = 1:length(regions1)
    %     ii
    rA = regions1{i1}(:);
    if (isempty(rA))
        continue
    end
    %     rA = find(rA);
    for i2 = 1:length(regions2)
        if (boxOverlaps(i1,i2) == 0)
            continue;
        end
        rB = regions2{i2}(:);
        if (isempty(rB))
            continue
        end
        int = rA & rB;
        ints(i1,i2) = nnz(int);
        if (ints(i1,i2) > 0)
            un = rA | rB;
            uns(i1,i2) = nnz(un);
        end
    end
    %     ovp = max(ovp,ovp');
end
ovp = ints./(uns+eps);
ovp(ints==0) = 0;


    function boxes  = getRegionBoxes(regions)
        
        boxes = zeros(length(regions),4);
        for iBox = 1:length(regions)
            [y,x] = find(regions{iBox});
            boxes(iBox,:) = [min(x,[],1) min(y,[],1) max(x,[],1) max(y,[],1)];
        end
    end
end