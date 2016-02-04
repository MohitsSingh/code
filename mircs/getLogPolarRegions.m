function r = getLogPolarRegions(sz,nTheta,theta_offset,theta_ovp,varargin)


ip = inputParser;
ip.addParameter('mask',[]);
ip.addParameter('center',[]);
ip.parse(varargin{:});
mask = ip.Results.mask;
center = ip.Results.center;
if isempty(center)
    center = sz([2 1])/2;
end
[X,Y] = meshgrid(1:sz(1),1:sz(2));
X = X-center(1);Y = Y-center(2);
R = (X.^2+Y.^2).^.5;
maxR = 45;
minR = 1;

if (~isempty(mask))
    R = bwdist(mask);
    %   imagesc(R)
    R1 = bwdist(~mask);
    %   figure,imagesc(-R1);
    
    mm = -R1.*mask+R.*(~mask);
    R = mm-min(mm(:));
end
theta = 180+atan2(X,Y)*180/pi;

if (nargin < 2)
    nTheta = 5;
end

min_theta = min(theta(:));
max_theta = max(theta(:));

debug_ = false;

if (nargin < 3 || isempty(theta_offset))
    theta_offset = (max_theta-min_theta)/(2*nTheta);
end

if (nargin < 4)
    theta_ovp = 10;
end

theta_range = theta_offset + (min_theta:(max_theta-min_theta)/(nTheta):max_theta);

theta_range = theta_range(1:end-1);
theta_range = mod(theta_range,max_theta);
r = {};
for iTheta = 1:length(theta_range)
    lo = theta_range(iTheta)-theta_ovp;
    hi_ind = iTheta+1;
    if (iTheta==length(theta_range))
        %hi = theta_range(1);
        hi_ind = 1;
    end
    hi = theta_range(hi_ind)+theta_ovp;
    
    if (hi > max_theta)
        hi = min_theta+mod(hi,max_theta);
    end
    
    if (hi < lo)
        curMask = (theta>=lo | theta <= hi);
    else
        curMask = (theta>lo & theta < hi);
    end
    r{iTheta} = curMask & R < maxR & R > minR;
    if (debug_)
        clf;
        subplot(1,2,1);imagesc(r{iTheta});
        subplot(1,2,2);imagesc(bsxfun(@times,im2double(r{iTheta}),im2double(I)));
        pause;
    end
end
