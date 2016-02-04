function ecs = ChainEdgels(I, verbose)

% chain edgels from edge-strength image I.
%
% Input:
% I(x,y) = 0 --> x,y is no edgel 
% I(x,y) > 0 --> x,y is an edgel of strength I(x,y)
% non-maxima suppression and hysteresis should have already been applied to I.
%
% Goal:
% produce edgel-chains at least as good as VxL's oxl chainer.
%
% Algorithm:
% The algorithm iteratively starts a new chain from the strongest edgel, and then does:
%  1. add the strongest 4-neighboring (up,down,left,right) edgel 
%     if no 4-neighbor exists goto 2
%  2. add the strongest diagonal-neighboring edgel
%     if none exists exists goto 3
%  3. add the strongest edgel at distance up to 2 (in the 5x5 neighborhood)
%     if none exists close the current chain and start a new one.
% If a the chain has been extended, move to the newly added edgel, clear it from I, and goto 1
%
% Chains with less than 4 edgels are removed
% (VxL's chainer also removes short chains, but more aggressively, needing higher length to survive)
%
% Output:
% ecs(i).chain(:,k) = [x y] coordinates of kth edgel of ith chain
%                     WARNING: they are in image coordinates, starting from (0,0),
%                     so as to be fully compatible with the original VxL's chainer
%                     (and as expected by routines to compute the edge strengths in Vitto's Matlab code)
%

% parse arguments
if nargin < 2
  verbose = false;
end

% add 2 pixel empty border to I,
% so that 5x5 neighborhood of any pixel is well defined
trash = zeros(size(I,1)+4,size(I,2)+4,'uint8');
trash(3:(2+size(I,1)),3:(2+size(I,2))) = I;
I = trash;
clear trash;

if verbose > 1
figure; imshow(I); hold on;
colormap jet;
end

% some useful shortcuts
i4 = [8 12 14 18];                                        % indeces of 4-neighb in 5x5 neighb
d4 = [ 0 -1 1 0                                           % displacements
      -1  0 0 1 ];
%
iD = [  7  9 17 19];                                      % indeces of 4-diag in 5x5 neighb
dD = [ -1  1 -1 1
       -1 -1  1 1];
%
iR = [    1:5          6 10  11 15  16 20     21:25];     % indeces of outer ring in 5x5 neighb
dR = [-2 -1 0   1  2  -2  2  -2  2  -2  2  -2 -1 0 1 2
      -2 -2 -2 -2 -2  -1 -1   0  0   1  1   2  2 2 2 2];

% find edgels and sort them according to their strengths
% WARNING: do not negate ss ! They are unit8 !
[xs ys ss] = find(I);
if isempty(xs)                      % no edgels in I
  disp([mfilename ': Warning: no edgels in image.']);
  ecs = [];
  return;
end
[trash ixs] = sort(ss,'descend');
ss = ss(ixs)';
xs = xs(ixs)';
ys = ys(ixs)';
Neds = length(ss);                  % number of edgels

% construct edgel-chains
ecs(1).chain = [];
start_ix = 1;             % start from strongest edgel
found = true;
E = [xs(1); ys(1)];       % current edgel
col = rand(1,3);
dir = 1;                  % forward run 
while found

  % add E to current edgel-chain
  ecs(length(ecs)).chain = [ecs(length(ecs)).chain E];
  I(E(1),E(2)) = 0;
  if verbose >1 
    plot(E(2),E(1),'o','color',col);
  end
  %
  % 5x5 neighborhood of E
  N5 = I( (E(1)-2):(E(1)+2), (E(2)-2):(E(2)+2) );
  %
  found = false;
  % is there a 4-neighboring edgel ?
  [val ix] = max(N5(i4));
  if val > 0
    E = E + d4(:,ix);
    found=true;
  else
    % is there a 4-diag edgel ?
    [val ix] = max(N5(iD));
    if val > 0
      E = E + dD(:,ix);
      found=true;
    else
      % is there an edgel on the outer ring ?
      [val ix] = max(N5(iR));
      if val > 0
        E = E + dR(:,ix);
        found=true;
      end
    end
  end
  %
  % deal with case in which no edgel to be added to the chain has been found
  if not(found)
    % need to run in the other direction ?
    if dir == 1
      ecs(length(ecs)).chain = ecs(length(ecs)).chain(:,size(ecs(length(ecs)).chain,2):-1:2);  % reverse chain and don't include starting edgel (will be readded soon ;)
      E = [xs(start_ix); ys(start_ix)];
      dir = 2;
      found = true;
    else
      % end of current chain
      found = false;
      while not(found) & start_ix < Neds
        start_ix = start_ix+1;
        found = (I(xs(start_ix),ys(start_ix))>0);
      end
      if found
        % start a new chain
        E = [xs(start_ix); ys(start_ix)];
        dir = 1;
        col = rand(1,3);
        ecs(length(ecs)+1).chain = [];
      end
      if verbose > 2  keyboard;  end
    end
  end

end

% invert x,y of the chains and remove 2 pixel offset,
% plus another pixel to make coord start at (0,0)
if verbose
  disp(['Found ' num2str(length(ecs)) ' chains.']);
end
for ecix = 1:length(ecs)
  ch = ecs(ecix).chain;
  ecs(ecix).chain = [ch(2,:)-3; ch(1,:)-3];
end

% remove chains with less than 4 edgels
keep = [];
for ecix = 1:length(ecs)
  if size(ecs(ecix).chain,2) > 3
    keep = [keep ecix];
  end
end
ecs = ecs(keep);
