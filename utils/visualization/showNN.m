function showNN(feats,imgs,knn,includeFirst,iu)
    if nargin < 3
        knn = 15;
    end
    if nargin < 4
        includeFirst = false;
    end                
    if nargin < 5
        D = l2(feats',feats');
%         D = -feats'*feats;
        [u,iu] = sort(D,2,'ascend');
    end
    figure(1);    
    for t = 1:1:length(imgs)
        t
        clf; subplot(1,2,1); imagesc2(imgs{t});        
        if (includeFirst)
            curNeighbors = iu(t,1:knn);
        else
            curNeighbors = iu(t,2:knn+1);
        end
        subplot(1,2,2); imagesc2(mImage(imgs(curNeighbors)));
        pause
    end
end