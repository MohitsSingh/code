function seg_probImage = superpixelize(I,prob_image,superPixMap,killBorders)
if (nargin < 3)
    I_ = single(vl_xyz2lab(vl_rgb2xyz(I)));
    superPixMap = vl_slic(I_,.1*size(I,1),1);
end
superPixMap = RemapLabels(superPixMap); % fix labels to range from 1 to n, otherwise a mex within constructGraph crashes.
% end

rprops = regionprops(superPixMap,prob_image,'PixelIdxList','MeanIntensity','Centroid');
seg_probImage = zeros(size(prob_image));
if (nargin < 4)
    killBorders = 0;
end
if (killBorders)
    borderImage = ones(size(prob_image));
    borderImage = addBorder(borderImage,killBorders,0);
    %     borderImage(killBorders+1:end-killBorders,killBorders+1:end-killBorders) = 0;
    borderPixels = find(borderImage);
end

for k = 1:length(rprops)
    curPixels = rprops(k).PixelIdxList;
    if (killBorders)
        if (~isempty(intersect(borderPixels,curPixels)))
            rprops(k).MeanIntensity = 0;
        end
    end
end

% imagesc(qq);
prob_unary = [rprops.MeanIntensity];
% prob_unary = normalise(prob_unary);

for k = 1:length(rprops)
    seg_probImage(rprops(k).PixelIdxList) = prob_unary(k);
end
