function [allSegments] = getSuperPix(VOCopts,trainIDs,superpixDir,regionSize,...
    regParam,ignoreSave,verbose)
%LEARNDICTIONARY Summary of this function goes here
%   Detailed explanation goes here
wasCell = true;
if (~iscell(trainIDs))
    trainIDs = {trainIDs};
    wasCell = false;
end

if (nargin < 6)
    ignoreSave = 0;
end
if (nargin < 5)
    regParam = .05;
end
if (nargin < 7) 
    verbose = false;
end
   

rr = num2str(regionSize);
gg = num2str(regParam);
for ii = 1:length(trainIDs)
    if (verbose)
        ii
    end
    currentID = trainIDs{ii};
    spPath = fullfile(superpixDir,[currentID '_' rr '_' gg '.mat']);
    if (exist(spPath,'file') && ~ignoreSave)
        load(spPath)
    else
        im = im2single(readImage(VOCopts,trainIDs{ii}));
        regularizer =regParam;%.05;
        segments = vl_slic(im,regionSize,regularizer);
        [segments] = RemapLabels(segments);
        %         [segImage] = paintSeg(im,segments);
        %         imshow(segImage);
        save(spPath,'segments');
    end
    allSegments{ii} = segments;
    %
    %     bowFeatures{ii} = index;
    %     bowFrames{ii} = frames;
end
if (~wasCell)
    allSegments = allSegments{1};
end
end