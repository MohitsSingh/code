function [bowFeatures,bowFrames] = getBows(VOCopts,trainIDs,dict,bowDir,verbose)
%LEARNDICTIONARY Summary of this function goes here
%   Detailed explanation goes here

phowOpts = {'Sizes', 4, 'Step', 5,'Color','RGB'};
bowFeatures = {};
bowFrames = {};
kdtree = vl_kdtreebuild(dict);
if (nargin < 5)
    verbose = true;
end
for ii = 1:length(trainIDs)
    if (verbose && mod(ii,50)==0)
        disp(ii)
    end
    currentID = trainIDs{ii};
    bowPath = fullfile(bowDir,[currentID '.mat']);
    if (exist(bowPath,'file'))
        load(bowPath)
    else
        im = im2single(readImage(VOCopts,trainIDs{ii}));
        [frames,descrs] = vl_phow(im, phowOpts{:},'FloatDescriptors',true); %#ok<*ASGLU>
        [index] = vl_kdtreequery(kdtree,dict,descrs); %#ok<*NASGU>
        save(bowPath,'frames','index');
    end
    
    bowFeatures{ii} = uint16(index); % assume dictionary size less that 2^16
    bowFrames{ii} = single(frames(1:2,:)); % no need for rotation and scale, only location in image
end
end