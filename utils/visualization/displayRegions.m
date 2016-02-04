function q = displayRegions(im,regions,scores,varargin)
% DISPLAYREGIONS displays regions one by one overlayed on an image of the 
% same size. By default, the function
% pauses after each region, unless the 'dontPause' parameter is specified.
%
% PARAMTERES:
% im - the image over which to display the regions
%   regions - a cell array of binary masks the same size as the first two
% dimensions as the image
%   scores - if not empty, regions will be shown according to descending
% order of this array, the same size as regions.
% optional arguments:
% delay : the delay after each display region, in seconds. 0 (default) means pause
% until keystroke. 
% dontPause : if set to true, overrides delay and does not pause after
% displaying each mask. Use together with "show"
% show: if set to false, will render the regions on the image but will not
% display the result. 
% See also showSorted
im = im2double(im);
if (isempty(regions))
    return;
end

if (nargin < 3 || isempty(scores))
    gotScores = false;
    iScore = 1:length(regions);
    scores = iScore;
else
    gotScores = true;
    [scores,iScore] = sort(scores,'descend');
end

ip = inputParser;
ip.addParameter('delay',0,@isscalar);
% ip.addParameter('regionScore',[],@isscalar);
ip.addParameter('maxRegions',inf,@isscalar);
ip.addParameter('moreFun',[]);
ip.addParameter('dontPause',false,@islogical);
ip.addParameter('show',true,@islogical);
ip.parse(varargin{:});
delay = ip.Results.delay;
maxRegions = ip.Results.maxRegions;
moreFun = ip.Results.moreFun;
dontPause= ip.Results.dontPause;
show = ip.Results.show;

if show
    figure(gcf); % bring up the current figure
end
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

maxRegions = min(maxRegions,length(regions));
if show
    cla;
end
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
    if show
        cla;
        imagesc2(normalise(q));
    end
    
    if (gotScores)
        scoreString = sprintf(': score= %3.3f',scores(ii));
    else
        scoreString = '';
    end
    titleString = {sprintf('%d out of %d%s',iScore(ii), length(regions),scoreString)};
    if (maxRegions ~= length(regions))
        titleString{2} = sprintf('(showing top %d)',maxRegions);
    end
    if show
        title(titleString);
    end
    if (~isempty(moreFun))
        feval(moreFun(z));
    end
    
    if ~dontPause
        if (delay == 0)
            disp('hit any key to continue to next region...');
            pause;
            
        elseif delay > 0
            pause(delay);
        end
    end
    if show
        drawnow
    end
end

if (nargout == 0)
    q = [];
end
end
