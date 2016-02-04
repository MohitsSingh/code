function [bestL] = applyGraphcut(currentImage,superPixMap,graphData)
% edge_param = 15;
pairwise_ = graphData.pairwise;
unary_ = graphData.unary;
labelcost = ones(2)-eye(2);
segclass = ones(size(unique(superPixMap(:))))';
rprops = regionprops(superPixMap,'PixelIdxList');


props = {};
for k = 1:3
    props{k} = regionprops(superPixMap,currentImage(:,:,k),'MeanIntensity');
end
colors = [[props{1}.MeanIntensity];[props{2}.MeanIntensity];[props{3}.MeanIntensity]]+eps;
C = squeeze(vl_xyz2luv(vl_rgb2xyz(reshape(colors', [size(colors,2), 1, size(colors,1)]))))';
colors = C';


bestL = zeros(size(currentImage,1),size(currentImage,2));
bestDiff = 0;
nDiffs = 0;
bestArea = 0;
for edge_param = 15:5:55
    
    [labels] = GCMex(double(segclass), single(unary_),...
        sparse(edge_param*pairwise_), single(labelcost));
    if (sum(labels)==0 || (sum(labels) > bestArea && bestArea > 0))
        continue;
    end
    nDiffs = nDiffs + 1;
    c_inside = colors(labels>0,:);
    c_outside = colors(labels==0,:);
    
    t  = l2(c_inside,c_outside).^.5;
    curDiff = mean(t(:));
    if (curDiff > bestDiff)
        bestDiff = curDiff;
        L1 = zeros(size(superPixMap));
        for k = 1:length(rprops)
            L1(rprops(k).PixelIdxList) = labels(k);
        end
        bestL = L1;
        bestArea = sum(labels);
%                 imshow(bestL,[]);title(num2str(edge_param));
%                 pause;
        %
    end
end

% L1 = bestL;
%     pause;
% end

end
