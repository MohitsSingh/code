function [scores_highres,t0] = doHighResScan(rgb_orig,boxes,patches,net_sub_patch)
imageNeedsToBeMultiple = true;
inputVar = 'input';
sz_orig = size2(rgb_orig);
softMaxEach = true;
scores_full_hires = [];% zeros([sz_orig,length(labels_local)]);
counts_hires = zeros(sz_orig);
marginSize = 10;
patches = patches(1:min(10,length(patches)));
patches = multiResize(patches,[384 384]);
% split patches to batches. no catches :-) 
% batches = batchify(1:length(patches),3);


% patches = cat(4,patches{:});
% patches = imResample(cat(4,curPatches{:}),,'bilinear');
% [~,scores_local] = ...
%         predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,patches,2,false);
tic
for iPatch = 1:min(10,length(patches))
%     iPatch
    curBox = boxes(iPatch,:);
    curPatch = patches{iPatch};
%     [~,scores_local] = ...
        scores_local = applyNet(net_sub_patch,curPatch,imageNeedsToBeMultiple,inputVar,'prediction');
        %predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,curPatch,2,false);
    if softMaxEach
        scores_local = bsxfun(@rdivide,exp(scores_local),sum(exp(scores_local),3));
    end
    scores_local_in_orig = transformBoxToImage(rgb_orig,scores_local,boxes(iPatch,:),false);
    curBox = curBox+marginSize*[1 1 -1 -1]; % avoid problems due to interpolation at edge.
    mask_in_orig = box2Region(curBox,sz_orig); %
    scores_local_in_orig = bsxfun(@times,scores_local_in_orig,mask_in_orig);
    if iPatch == 1
        scores_full_hires = scores_local_in_orig;
    else
        scores_full_hires = scores_full_hires+scores_local_in_orig;        
    end
    counts_hires = counts_hires+mask_in_orig;
    %         dpc
end
t0 = toc;
scores_highres = bsxfun(@rdivide,scores_full_hires,counts_hires+eps);
if ~softMaxEach
    scores_highres = bsxfun(@rdivide,exp(scores_highres),sum(exp(scores_highres),3));
    scores_highres = bsxfun(@times,scores_highres,counts_hires);
end