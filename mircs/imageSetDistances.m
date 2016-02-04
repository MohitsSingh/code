function [dists12,A12] = imageSetDistances(conf,A,B)
% calculates minimal distances between all features pairs.
% For two set of images, A, B, returns a cell array dists12 of size length(A) X length (B)
% where dists12{i}{j} contains for each feature Fa in image A{i}
% the distance to the nearest feature Fb in image A{j}.
% So cat(1,dists12{i}{:}) is a m x k matrix specifying
% for each feature F1....Fk in image A{i} the distance to the nearest feature
% in each of the m images in B.
% A12 corresponding features's indices in the neighboring image.
dists12 = {};
A12 = {};
k = 1; % number of nn to find (negative for furthest)

parfor ii = 1:length(A)
    disp(['calculating descriptors for first set: %' num2str(100*ii/length(A))]);
    if (ischar(A{ii}))
        I1 = toImage(conf,getImagePath(conf,A{ii}));
    else
        I1 = A{ii};
    end
    [X1,uus1,vvs1,scales1,~] = allFeatures(conf,I1,.3);
%     X1 = normalize_vec(X1);
%     [X1] = sampleHogs(conf,{I1},'',inf,0);
    x1{ii} = single(X1);    
end

parfor ii = 1:length(B)
    disp(['calculating descriptors for second set: %' num2str(100*ii/length(B))]);
       if (ischar(B{ii}))
        I2 = toImage(conf,getImagePath(conf,B{ii}));
       else
             I2 = B{ii};
       end
%     [X2] = sampleHogs(conf,{I2},'',inf,0);
    [X2] = allFeatures(conf,I2,.3);
%     X2 = normalize_vec(X1);
    x2{ii} = single(X2);
end

parfor ii = 1:length(A)
    %%
    if ~isempty(x1{ii})
        disp(['finding nn for images: %' num2str(100*ii/length(A))]);
        for jj = 1:length(B)
            [d12,~,a12] = getNN(x1{ii}',x2{jj}',k);
            dists12{ii}{jj} = d12;
            A12{ii}{jj} = a12;
        end
    end
end