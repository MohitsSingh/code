function restricted_patches = sample_restricted(imgs,myParam)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    mouth_corner_inds = [35 41];
    restricted_patches = {};
    for t = 1:length(imgs)   
        if (size(imgs(t).landmarks_local,1)==39)
            warning('sample_restricted====>skipping profile face!')
            continue;
        end
        curMouthPts = imgs(t).landmarks_local(mouth_corner_inds,:);
        I = imgs(t).I;
        bb = anchoredBoxSamples(I,curMouthPts,myParam);
        restricted_patches{end+1} = multiCrop2(I,bb);
    end
    restricted_patches = cat(2,restricted_patches{:});