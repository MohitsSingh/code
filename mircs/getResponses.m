function [r,boxes] = getResponses(conf,clusters,I) % apply multiple identical sized detector to image.
    [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,I,1 );
    boxes = boxes(:,1:4);
    exemplar_matrix = cat(2,clusters.w);
    bs = cat(1,clusters.b);
    r = exemplar_matrix' * X;
    r = bsxfun(@minus, r, bs);
end