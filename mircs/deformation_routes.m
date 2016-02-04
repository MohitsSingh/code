lipImages_train1 = lipImages_train;(train_faces_scores_r>=-.4);
lipImages_train1 = train_faces;
% mImage(lipImages_train1);

winSize = [12 12];
dwinSize = winSize*8;
conf.features.winsize = winSize;
conf.features.vlfeat.cellsize = 8;

x = imageSetFeatures2(conf,lipImages_train1,true,dwinSize);

D = l2(x',x').^.5;

D(D<10-9) = 0;

%n = randperm(length(lipImages_train1));

vals = D(D(:)>0);
vals = normalise(vals)+.000001;
vals(vals>1) = 1;
D(D(:)>0) = vals;


%%
%T = .4;

[R,IR] = sort(D,2,'ascend');

knn = 10;
ii = repmat((1:size(D,1))',knn,1);
jj = IR(:,2:2+knn-1);
jj = jj(:);
ss = D(sub2ind(size(D),ii,jj)).^4;
%G = sparse(ii,jj,ss,size(D,1),size(D,2));
G = sparse(D.^4);

%G = sparse(D.*(D<=T));

% figure,imagesc(sort(D,2,'ascend'));
S = 1;
%%
S = f(4);
[dist, path, pred] = graphshortestpath(G,S,'Directed',false);
% dists = graphallshortestpaths(G,'Directed',false);
% figure,imagesc(dists)
% [r,ir] = sort(dists,2,'ascend');
% figure,imagesc(r)
% [ii,jj] = find(r==max(r(~isinf(r(:)))));
ims = lipImages_train1;

% S = 1;
% % T = 868;
% [dist, path, pred] = graphshortestpath(G,S,'Directed',false);
% 

lengths = cellfun(@length,path);
% lengths = length(path);
%%
[L,iL] = sort(lengths,'descend');
L(1)

[L,iL] = sort(dist,'descend');

% path = {path};
for iPath = 1:length(path)
    
%     if (~t_train(iL(iPath)))
%         continue;
%     end
    if (lengths(iL(iPath)) < 2)
        continue;
    end
        iPath
    clf;
    curPath
    
    if (~t_train(iL(iPath)))
        continue;
    end
    
    subplot(1,3,1);
    imagesc(ims{S}); axis image; title('S');
    curPath = path{iL(iPath)}; 
    subplot(1,3,3);
    imagesc(ims{curPath(end)}); axis image; title('T');
    for iNode = 1:length(curPath);
        subplot(1,3,2);
        imagesc(ims{curPath(iNode)}); axis image; title(num2str(iNode));
        if (iNode<length(curPath))
            pause(.1);
        end
    end
    pause; title(['done (' num2str(length(curPath)) ')']);

end

