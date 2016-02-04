function becs = BinEdgelChains(ecs, Nw, Nh)

% Divides the area covered by edgelchains ecs
% in a grid of Nw x Nh rectangular bins,
% then computes which edgel chains pass through which bin.
% This is useful to speedup later algorithms.
%
% Input:
% ecs(c).chain(:,n) = [x;y] coordinates of nth pixel of cth chain
%
% Output:
% becs.range = [minx maxx miny maxy] = range of area covered by edglechains
%              (useful to reconstruct bins' locations)
% becs.B(i,j).ecix = set of chains passing through bin i,j
%

if nargin < 3
  Nw = 20; Nh = 20;                       % default: 20x20 bins
end

% Compute range
minx = 0;
maxx = 0;
miny = 0;
maxy = 0;
for ecix = 1:length(ecs)
  ec = ecs(ecix).chain;
  minx = min([minx min(ec(1,:))]);
  maxx = max([maxx max(ec(1,:))]); 
  miny = min([miny min(ec(2,:))]);
  maxy = max([maxy max(ec(2,:))]);
end
becs.range = [minx maxx+1 miny maxy+1];   % '+1' -> make sure a point with x=maxx, or y=maxy will be inside the last bin
bw = (maxx+1-minx)/Nw;                    % bin width
bh = (maxy+1-miny)/Nh;                    % bin height

% Compute binning
B(Nw*Nh).ecix = [];                       % allocate memory
for ecix = 1:length(ecs)
  % Current chain
  ec = ecs(ecix).chain;

  % Coordinates-to-bin equation:
  % 1+floor((x-minxx)/bw)   1+floor((y-miny)/bh)
  bins = [1+floor((ec(1,:)-minx)/bw); 1+floor((ec(2,:)-miny)/bh)];

  % Convert bin to [1 Nw*Nh] to speedup assignments
  bins = bins(1,:) + (bins(2,:)-1)*Nw;
  bins = unique(bins);
  
  % Update bins' contents
  for b = bins
     B(b).ecix = [B(b).ecix ecix];
  end
end

% Convert binning to bin_x,bin_y format
for b = 1:Nw*Nh
  by = floor((b-1)/Nw)+1;
  bx = b-(by-1)*Nw;
  becs.B(bx,by).ecix = B(b).ecix;
end
