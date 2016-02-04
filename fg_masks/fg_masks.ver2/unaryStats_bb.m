function [ Pw ] = unaryStats_bb(VOCopts,trainIDs,bowFeatures,...
    bowFrames,dict,recs,fg_masks)
%UNARYSTATS Summary of this function goes here
%   Detailed explanation goes here
% count for each visual word how many times it appeared as bg / fg
bg = zeros(size(dict,2),1);
fg = zeros(size(dict,2),1);
Pw = zeros(size(bg));
independent_stats =0;


for k = 1:length(trainIDs)
    
    if (nargin < 7)
        if (nargin >= 6)
            fg_mask = pasRec2Mask(VOCopts,trainIDs{k},recs{k});
        else
            fg_mask = pasRec2Mask(VOCopts,trainIDs{k});
        end
    else
        % foreground mask for this image was supplied externally,
        %so use it instead. Assume no "don't care" area in image.
        fg_mask = fg_masks{k};
    end
    fr = bowFrames{k};
    feat = bowFeatures{k}(:);
    inds = sub2ind(size(fg_mask),fr(2,:),fr(1,:));
    
    % a weighted sum for each feature according to the given maks
    subs_ = [feat,ones(length(inds),1)];
    fg_weighted = accumarray(subs_,fg_mask(inds),[size(dict,2) 1]);
    total_weighted = accumarray(subs_,ones(size(inds)),[size(dict,2) 1]);
    bg_weighted = total_weighted-fg_weighted;
    
    if (independent_stats)
        Pw = Pw + length(feat)*(fg_weighted./(total_weighted+eps));
    else
        bg = bg+bg_weighted;
        fg = fg+fg_weighted;
    end
end
if (~independent_stats)
    Pw = fg./(bg+fg+eps);
else
    Pw = Pw/sum(Pw);
end
end

