function [cg,tg,theta] = detCGTG(im,radius,norient)
% function [cg,tg,theta] = detCGTG(im,radius,norient)
%
% Compute smoothed but not thinned CG and TG fields.

if nargin<2, radius=[0.01 0.02 0.02 0.02]; end
if nargin<3, norient=8; end

if numel(radius)==1, radius = radius*ones(4,1); end

[h,w,unused] = size(im);
idiag = norm([h w]);

% compute color gradient
[cg,theta] = cgmo(im,idiag*radius(1:3),norient,...
                  'smooth','savgol','sigmaSmo',idiag*radius(1:3));

% compute texture gradient
no = 6;
ss = 1;
ns = 2;
sc = sqrt(2);
el = 2;
k = 64;
fname = sprintf('unitex_%.2g_%.2g_%.2g_%.2g_%.2g_%d.mat',no,ss,ns,sc,el,k);
% original code: these lines crash on matlab v7 due to name clash with the tex toolbox
%load(fname); % defines fb,tex,tsim
%tmap = assignTextons(fbRun(fb,rgb2gray(im)),tex);
%
% update to matlab v7 by V. Ferrari
stuff = load(fname);
tmap = assignTextons(fbRun(stuff.fb,rgb2gray(im)),stuff.tex);

[tg,theta] = tgmo(tmap,k,idiag*radius(4),norient,...
                  'smooth','savgol','sigma',idiag*radius(4));

