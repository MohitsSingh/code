function [regions,regionsInds,imageInds,allProps] = extractProperties(images,ucms,qq_det,suffix)

resPath = fullfile('~/storage',[suffix '_props.mat']);
if (exist(resPath,'file'))
    load(resPath);
    return;
end

debug_ = false;
traceBoundaries = false;
if (~debug_)
    results = struct('imageIndex',{},'maxScore',{},'candidates',{},'scores',{});
end
resultCount = 0;
scaleFactor = 1;
Z_prior = false(100);
Z_prior(40:70,30:70) = 1;
Z_prior = imresize(Z_prior,scaleFactor,'nearest');
bb = single(cat(1,qq_det.cluster_locs));
mm =2;
nn = 3;

regions = {};
regionsInds = {};
imageInds = {};
allProps = {};
nRegions = 0;

% list of properties...
propertyList = {'MajorAxisLength','MinorAxisLength','PixelIdxList',...
    'Solidity','BoundingBox','PixelList','Orientation','Eccentricity','Area','MeanIntensity'};


for k = 1:length(images)
    k
    curIm = imresize(images{k},2*scaleFactor,'bilinear');
    curIm_gray = im2double(rgb2gray(curIm));
    ucm = ucms{k};
    ucms{k} = [];
    %ucm = normalise(ucm);
    rprops = [];
    if (debug_)
        clf;
        %for kk = [.1 .2];
        qqqq = 1;
        figure(2); clf;
        subplot(3,4,1);
        imagesc(curIm);axis image;
    end
    
    
    propsFileName = fullfile('~/storage/props',sprintf('%s_%05.0f.mat',suffix,k));
    if (exist(propsFileName,'file'))
        load(propsFileName);
    else
        
        %     for kk = .01
        for kk = [.05:.1:.3]
            L = bwlabel(ucm <= kk);
            if (debug_)
                segImage = paintSeg(curIm,L);
                kk
                qqqq = qqqq+1;
                subplot(3,4,qqqq);imagesc(segImage); axis image;
            end                                               
            %         curProps = regionprops(L,curIm_gray,'MajorAxisLength','MinorAxisLength','PixelIdxList',...
            %             'Solidity','BoundingBox','PixelList','Orientation','Eccentricity','Area','MeanIntensity');
            curProps = regionprops(L,curIm_gray,propertyList(:));
            for ii = 1:length(curProps)
                curProps(ii).kk = kk;
            end
            %[curProps.kk] = repmat(kk,length(curProps),1);
            if traceBoundaries
                B = bwboundaries(L);
                for iBoundary = 1:length(curProps)
                    curProps(iBoundary).perim = B{iBoundary};
                end
            end
            rprops = [rprops;curProps];
        end
        
        save(propsFileName,'rprops')
    end
    
    curBoxes = bb(bb(:,11)==k,:);
    vals = normalise(curBoxes(:,12));
    sz = size(curIm);
    Z = zeros(sz(1:2));
    Z = drawBoxes(Z,curBoxes,vals,2);
    Z = Z/max(Z(:));
    
    for p = 1:length(rprops)
        curPts = rprops(p).PixelList;        
        [x,ix]=  sort(curPts(:,2),'ascend');
        nRegions = nRegions+1;
        minPts = single(curPts(ix(1),:));
        maxPts = single(curPts(ix(end),:));
        rprops(p).minPts = minPts;
        rprops(p).maxPts = maxPts;
        rprops(p).startPts = single(Z_prior(minPts(2),minPts(1)));
        rprops(p).z_in = single(Z(minPts(2),minPts(1)));
        rprops(p).z_out = single(Z(maxPts(2),maxPts(1)));
        regions{end+1} = uint16(rprops(p).PixelIdxList);
        imageInds{end+1} = uint16(ones(size(rprops(p).PixelIdxList))*k);
        regionsInds{end+1} = uint32(ones(size(rprops(p).PixelIdxList))*nRegions);
    end
    
    rprops = rmfield(rprops,'PixelList');
    rprops = rmfield(rprops,'PixelIdxList');
    rprops = rmfield(rprops,'BoundingBox');
    
    allProps{end+1} = rprops;
    
    %     drawnow;
end

allProps = cat(1,allProps{:});

% 1 area
% 2 ecce
% 3 kk
% 4 maj. length
% 5 max. pts. x
% 6 max. pts. y
% 7 min. pts. x
% 8 min. pts. y
% 9 mean intensity
% 10 minor axis
% 11 orientation
% 12 solidity
% 13 startPts
% 14 z_in
% 15 z_out

allProps = [[allProps.Area];[allProps.Eccentricity];[allProps.kk];[allProps.MajorAxisLength];...
    cat(1,allProps.maxPts)';cat(1,allProps.minPts)';[allProps.MeanIntensity];[allProps.MinorAxisLength];...
    [allProps.Orientation];[allProps.Solidity];[allProps.startPts];[allProps.z_in];[allProps.z_out]];

regions = uint16(cat(1,regions{:}));
regionsInds = uint32(cat(1,regionsInds{:}));
imageInds = uint16(cat(1,imageInds{:}));

save(resPath,'regions','regionsInds','allProps','imageInds');




