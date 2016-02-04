function colors = meanSegColors( labels,I,rgbSpace)
%MEANSEGCOLORS Summary of this function goes here
%   Detailed explanation goes here
props = {};
if (nargin < 3)
    rgbSpace = true;
end

if (islogical(labels)) % one segment
    colors = zeros(3,1);
    for k = 1:3
        curChn = I(:,:,k);
        colors(k) = mean(curChn(labels));
    end
else
    
    for k = 1:3
        props{k} = regionprops(labels,I(:,:,k),'MeanIntensity');
    end
    
    colors = [[props{1}.MeanIntensity];[props{2}.MeanIntensity];[props{3}.MeanIntensity]]+eps;
    
    
end
if (~rgbSpace)
    colors = squeeze(vl_xyz2lab(vl_rgb2xyz(reshape(colors', [size(colors,2), 1, size(colors,1)]))))';
end

end

