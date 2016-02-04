function [vm, cntVote, idxVote, posXY, prob] = star_model_test(encoder, sm, img, params) 
    args_kdtreequery = {'NumNeighbors', params.k, 'MaxComparisons', params.k*100};

    features = params.extractorFn(img);

    descr = features.descr';
    posXY = features.frame(1:2,:)';

    % find k NN for one of the densly sampled patchs in the image

    % get votes and distance (size k x N)
    [idxVote, dst_sqr] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
                                descr', args_kdtreequery{:}); 
                        

    %calc probability by similarity distance 
    prob=exp(-0.5*dst_sqr/(params.sigma^2));

    %normalize such that the sum of prob of k NN is 1
    prob=bsxfun(@rdivide,prob,sum(prob)); 

    %calculate the center by voting
    clear cntVote;

    %for each patch position add the k NN offsets
    cntVote(:,:,1)=bsxfun(@plus,reshape(sm.offs2cntXY(idxVote,1),size(idxVote)),posXY(:,1)');
    cntVote(:,:,2)=bsxfun(@plus,reshape(sm.offs2cntXY(idxVote,2),size(idxVote)),posXY(:,2)');
    
    v=round(reshape(cntVote,[],2));

    %some votes might give illegal position of out side the image, remove them 
    vv=all(v>=1,2) & all(bsxfun(@le,v,fliplr(size(img))),2);

    %accumulate the probability of the valid votes of the positions to get heat map
    vm=accumarray(v(vv,[2 1]),prob(vv),size(img));

    % smooth with a gaussian filter
    vm=imfilter(vm,fspecial('gaussian',round(3*params.geom_sigma_cnt),params.geom_sigma_cnt));
    
return;
