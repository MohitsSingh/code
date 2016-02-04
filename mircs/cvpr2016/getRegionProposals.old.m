function [props,regions] = getRegionProposals(curPreds,softMaxScores,I,params)
useRawScores=params.useRawScores;
useLocalMaxima=params.useLocalMaxima;
p = curPreds;
p(p<=3) = 0;
regions = {};
nLabels = size(softMaxScores,3);
props = [];
M = max(softMaxScores(:,:,2:end),[],3);
S = max(softMaxScores(:,:,4:end),[],3);

if params.usePrediction
    for u = 4:nLabels
        curLabels = p == u;
        curLabels = imclose(curLabels,ones(10));
        %     if (any(curLabels(:))),continue,end
        curProps = [];
        if any(curLabels(:))
            pp = regionprops(curLabels,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
            curProps = [curProps;pp];
        end
        if useRawScores
            II = softMaxScores(:,:,u);
            [level,em] = graythresh(II);
            II = II>level;
            if any(II(:)) && ~all(II)
                curProps = [curProps;regionprops(II,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
            end
        end
        props = [props;curProps];
    end
end

if params.useMaxObj
    II = S > .1;
    [thresh,metric] = multithresh(S,1);
    U = imquantize(S,thresh);
    props = [props;regionprops(U>=2,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
    %props = [props;regionprops(U>=3,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
end

if useLocalMaxima
    [subs,vals] = nonMaxSupr( double(M), 20, .3,10 );
    [thresh,metric] = multithresh(M,2);
    U = imquantize(M,thresh);
    props = [props;regionprops(U==3,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList')];
    % obtain predictions and shapes around these local maxima
end

for u = 1:length(props)
    props(u).is_gt_region = false;
end