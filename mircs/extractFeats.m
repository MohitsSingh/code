function [feats,labels] = extractFeats(all_data,featureExtractor)
feats = struct('name',{},'feats',{});
labels = {};
app_patches = {};
mask_patches = {};
for t = 1:length(all_data)
    curImg = all_data(t).img;
    curRegions = all_data(t).regions;    
    lengthBefore = length(curRegions);
    [curRegions,selection] = ezRemove(curRegions,curImg,50,.3);
    if isempty(all_data(t).labels)
        all_data(t).labels = ones(size(curRegions));
    end
    curLabels = all_data(t).labels(selection);
    curLabels(curLabels==1) = all_data(t).classID;
    if length(curRegions) < lengthBefore
        disp(t)
    end
    for u = 1:length(curRegions)
        app_patches{end+1} = maskedPatch(curImg,curRegions{u},true,.5);
        mask_patches{end+1} = repmat(curRegions{u},[1 1 3]);
    end
   % assert(length(curLabels)==length(curRegions))
    
    %if nargin == 3 && all_data(t).classID ~= pos_class
        %labels{end+1} = -1*ones(size(curLabels));
    %else
    labels{end+1} = curLabels;
%     end
end
labels = cat(2,labels{:});
feats(1).name = 'patch appearance';
% for t = 1:length(app_patches)
%     t
%     featureExtractor.extractFeaturesMulti(app_patches(t));
% end
feats(1).feats = gather(featureExtractor.extractFeaturesMulti(app_patches));
feats(2).name = 'patch shape (global)';
feats(2).feats = gather(featureExtractor.extractFeaturesMulti(mask_patches));
fprintf('\n');
% shapeFeats = featureExtractor.extractFeaturesMulti(mask_patches);
end