function [X,locs] = samplePatches(conf,ids,ovp)
% Sample a semi-dense set of patches from each of the input images.
% The patches will have an (approximate) overlap defined by ovp or overlap
% of 1/3 (along each axis) if none is specified. Note - Overlap is measured only
% inside a single scale, so patches of different scales might have more
% overlap.
if (nargin < 3)
    ovp = 1/3;
end
X = {};
locs = {};
for k = 1:length(ids)
    disp(['calculating descs for image ' num2str(k) ' out of ' num2str(length(ids))]);
    I = getImage(conf,ids{k});
    [X_,uus,vvs,scales,t ] = allFeatures( conf,I,ovp);
    X{k} = X_;
    if (nargout == 2)
        locs_ = uv2boxes( conf,uus,vvs,scales,t );
        if (~isempty(locs_))
            locs_(:,11) = k;
            locs{k} = locs_;
        end
        
%         figure,imshow(I);
%         hold on;
%         plotBoxes2(locs_(:,[2 1 4 3]));
    end
end

X = cat(2,X{:});
if (nargout == 2)
    locs = cat(1,locs{:});
end