function q = displayRegions(im,regions,regionScore,delay,maxRegions,moreFun)
% displayRegions(im,regions,regionScore,delay,maxRegions)
im = im2double(im);
if (isempty(regions))
    return;
end
figure(gcf); % bring up the current figure
if (~iscell(regions)) % just a single region, or a stack of them
    if size(regions,3)>1
        r = regions;
        regions = {};
        for t = 1:size(r,3)
            regions{t} = r(:,:,t);
        end
    else
        regions = {regions};
    end
end
if (nargin < 4 || isempty(delay))
    delay = 0;
end
if (nargin < 5 || isempty(maxRegions))
    maxRegions = length(regions);
end

maxRegions = min(maxRegions,length(regions));
if (maxRegions<=1 && nargin < 4)
    delay = -1;
end
gotScores = true;
if (nargin < 3 || isempty(regionScore))
    gotScores = false;
    iScore = 1:length(regions);
    regionScore = iScore;
else
    [regionScore,iScore] = sort(regionScore,'descend');
end

alpha_ = .5;
cla;
% if (nargin < 6)
%         z = zeros(size(regions{1}));
% end
for ii = 1:maxRegions
    %     ii
    if (~iscell(regions))
        b = regions(iScore(ii),:);
        z = computeHeatMap(im,[b 1]);
    else
        z = regions{iScore(ii)};
    end
    z = fillRegionGaps({z});
    z = z{1};
    q = blendRegion(im,double(z),.3);
    cla;
    imagesc2(normalise(q));
    
    %     hold on;
    %title(num2str(sum(regions{iScore(ii)}(:))));
    gotScoreString = '';
    if (gotScores)
        gotScoreString = sprintf(': score= %3.3f',regionScore(ii));
    end
    titleString = {sprintf('%d out of %d%s',iScore(ii), length(regions),gotScoreString)};
    if (maxRegions ~= length(regions))
        titleString{2} = sprintf('(showing top %d)',maxRegions);
    end
    title(titleString);    
    if (nargin == 6)
        feval(moreFun(z));
    end    
    if (delay == 0)
        disp('hit any key to continue to next region...');
        pause;
        
    elseif delay > 0
        pause(delay);
    end
    
    drawnow
end

if (nargout == 0)
    q = [];
end
end
