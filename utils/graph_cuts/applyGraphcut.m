function [labels,labelImage] = applyGraphcut(superPixMap,graphData,edgeParam)

pairwise_ = graphData.pairwise;
unary_ = graphData.unary;
labelcost = ones(2)-eye(2);
segclass = ones(size(unique(superPixMap(:))))';
pairwise_ = pairwise_{1}.*pairwise_{2};%.*pairwise_{3};
% pairwise_ = pairwise_{2};

[labels] = GCMex(double(segclass), single(unary_),...
    sparse(edgeParam*pairwise_), single(labelcost));


% [labels] = GCMex(double(segclass), single(unary_),...
%     sparse(edgeParam*pairwise_), single(labelcost));

if (nargout == 2)
    rprops = regionprops(superPixMap,'PixelIdxList');
    labelImage = zeros(size(superPixMap));
    for k = 1:length(labels)
        labelImage(rprops(k).PixelIdxList) = labels(k);
    end
end


end
