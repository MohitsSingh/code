function fb2 = makeDoubleFilterBank(thetaLimits,dTheta,widths)

if (nargin < 1)
    thetaLimits = [0 170];
end
if (nargin < 2)
    dTheta = 10;
end
if (nargin < 3)
    widths = 1:4;
end

thetaRange = thetaLimits(1):dTheta:thetaLimits(2);
fb = FbMake(2,4,0);
fb = squeeze(fb(:,:,1));
b = zeros(dsize(fb,1:2));
b(1:end,ceil(size(b,2)/2)) = 1;
fb2 = zeros([size(fb) length(thetaRange)]);
b = b';
allFilters = {};
for iWidth = 1:length(widths)     
    baseFilter = conv2( circshift(b,[widths(iWidth) 0]), fb, 'same');    
    baseFilter = baseFilter + flipud(baseFilter);
    
    for k = 1:length(thetaRange)
        curFilter = imrotate(baseFilter,thetaRange(k),'bicubic','crop');
%         if (sum(curFilter > .01))
% %             curFilter = curFilter/ sum(curFilter(:));
%         end
        allFilters{end+1} = curFilter / sum(curFilter(:)+.1);
    end
end
fb2 = cat(3,allFilters{:});
figure,montage2(fb2);
