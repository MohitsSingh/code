function [props,regions] = getRegionProposals(curPreds,scores,I,params,bbox)
useRawScores=params.useRawScores;
useLocalMaxima=params.useLocalMaxima;
p = curPreds;
p(p<=1) = 0;
regions = {};
nLabels = size(scores,3);
props = [];
M = max(scores(:,:,2:end),[],3);
S = max(scores(:,:,4:end),[],3);
if params.usePrediction
    for u = 4:nLabels
        curLabels = p == u;
        curLabels = imclose(curLabels,ones(10));
        %     if (any(curLabels(:))),continue,end
        curProps = [];
        if any(curLabels(:))
            pp = regionprops(curLabels,M,'Area','MaxIntensity','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
            %             displayRegions(I,propsToRegions(pp,size2(I)));
            curProps = [curProps;pp];
        end
        if useRawScores
            II = scores(:,:,u);
            [level,em] = graythresh(II);
            II = II>level;
            if any(II(:)) && ~all(II)
                curProps = [curProps;regionprops(II,M,'Area','MaxIntensity','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
            end
        end
        props = [props;curProps];
    end
end

if params.useMaxObj
    %     II = S > .1;
    [thresh,metric] = multithresh(M,2);
    U = imquantize(M,thresh);
    
    curProps = regionprops(imclose(U>=3,ones(3)),M,'Area','MaxIntensity','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
    %     curProps = regionprops(U>=3,M,'Area','MaxIntensity','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
    props = [props;curProps];
    %props = [props;regionprops(U>=3,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
end

if useLocalMaxima
    for ii = 4:size(scores,3)
        S = double(scores(:,:,ii));
        [subs,vals] = nonMaxSupr( S, 5, 0,10);
        [thresh,metric] = multithresh(S,2);
        U = imquantize(S,thresh);
        curProps = regionprops(imclose(U>=3,ones(3)),S,'Area','MaxIntensity','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
        % select only regions that contain one of the values....
        subs_1 = sub2ind2(size(S),subs);
        members = false(size(curProps));
        for t = 1:length(curProps)
            m = ismember(subs_1,curProps(t).PixelIdxList);
            if any(ismember(subs_1,curProps(t).PixelIdxList))
                members(t) = true;
            end
        end
        curProps = curProps(members);
        props = [props;curProps];
%         clf,imagesc2(U);plotPolygons(fliplr(subs),'r+');dpc
    end
    %
    
    %
    %     [thresh,metric] = multithresh(M,2);
    %     U = imquantize(M,thresh);
    %     props = [props;regionprops(U==3,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
    % obtain predictions and shapes around these local maxima
end

for u = 1:length(props)
    props(u).is_gt_region = false;
end