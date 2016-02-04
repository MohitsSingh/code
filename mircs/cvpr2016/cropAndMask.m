function [patches,patches_prob,patches_hard_mask,patches_soft_mask,patches_seg_mask] = cropAndMask(I,S,boxes,avg_img)
S_orig = S;
patches = {};
patches_prob = {};
patches_hard_mask = {};
patches_soft_mask = {};
patches_seg_mask = {};
for u = 1:size(boxes,1)
    curBox = boxes(u,:);
    S = S_orig.*single(box2Region(curBox,size2(I)));
    S_crop = cropper(S,curBox);
    S_crop = normalise(S_crop);
    S_crop = S_crop.^.5;
%     [xx,yy] = meshgrid(1:size(S,1),1:size(S,2));
    %
    patches{u} = cropper(I,curBox);
    patches_prob{u} = S_crop;
    %
    curPatch = patches{u};
    curProb = patches_prob{u};
    %     curProb = cat(3,curProb,curProb,curProb);
    avg_img_p = imResample(avg_img,size2(curPatch),'nearest');
    patch_soft = bsxfun(@times,(1-curProb),avg_img_p)+bsxfun(@times,curProb,single(curPatch));
    curMask = single(curProb>=.5);
    patch_hard = bsxfun(@times,(1-curMask),avg_img_p)+bsxfun(@times,curMask,single(curPatch));
    patches_hard_mask{u} = uint8(patch_hard);
    patches_soft_mask{u} = uint8(patch_soft);
    bw1 = curProb > .5;
    bw2 = curProb < .2;
    bw2 = addBorder(bw2,1,1);
    bw2 = bw2 & ~bw1;
    [L, P] = imseggeodesic(curPatch,bw1,bw2,'AdaptiveChannelWeighting',false);
    L = single(L==1);
    patch_soft2 = bsxfun(@times,1-L,avg_img_p)+bsxfun(@times,L,single(curPatch));
    patches_seg_mask{u} = uint8(patch_soft2);
end