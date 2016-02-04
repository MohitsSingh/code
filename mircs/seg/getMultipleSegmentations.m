function labels = getMultipleSegmentations(im,kThresholds)
colorTypes = {'Rgb', 'Hsv','Lab'};%, 'RGI', 'Opp'};

% colorTypes = {'Lab'};%, 'RGI', 'Opp'};

if (nargin < 2)
    kThresholds = [10 20];
end
% kThresholds = 30;
sigma_=.5;
minSize = 15;
c = 0;
for iColor = 1:length(colorTypes)
    switch colorTypes{iColor}
        case 'Rgb'
            curIm = im;            
        case 'Hsv'
            curIm = im2uint8(rgb2hsv(im));
        case 'Lab'            
            curIm = uint8(rgb2lab(im));
    end        
%     curIm
    for iThresh = 1:length(kThresholds)
        c = c+1;        
        labels{c} = mexFelzenSegmentIndex(curIm, sigma_, minSize, kThresholds(iThresh));
        [num2str(kThresholds(iThresh) ) ' ' colorTypes{iColor}]
%         imagesc(labels{c});
%         pause;
    end
end

end