function sub_cats = makeSubCategories(pos_samples)
    X = reshape(pos_samples,[],size(pos_samples,4));
    [IDX,C] = kmeans2(X',10,struct('nTrial',100,'display',1));
    
    close all;display_samples(pos_samples(:,:,:,IDX==4))
       
    [clusters,ims_,inds] = makeClusterImages(pos_patches,C',IDX',X,[],100);
        
    for k = 1:size(C,1)
        display_samples(reshape(C(k,:),dsize(pos_samples,1:3)));
        pause
    end
    
    
    
    maxPerCluster = 300;
end