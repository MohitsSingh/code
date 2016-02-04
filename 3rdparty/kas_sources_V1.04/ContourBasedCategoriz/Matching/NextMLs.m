function [tot_nixs, tot_nepts, tot_depths] = NextMLs(mlix, mlnet, dir, used, depth_left, cur_depth)

% List of main-lines indexes that come
% 'next' after mlix in direction dir
% (dir==1==back, dir==2==front).
%
% if 'used' is provided, return only
% unused main-lines.
%
% if 'depth_left' is provided, follow the network and
% return all next main-lines up to that depth.
%

tot_nixs = [];
tot_nepts = [];
tot_depths = [];
nixs = [];
nepts = [];
depths = [];

if nargin < 6
  cur_depth = 1;
end

if nargin < 5
  depth_left = 1;
end

if depth_left == 0
  return;
end

if nargin < 4
  used = zeros(1,length(mlnet));
end

if dir == 2
  t = mlnet(mlix).front;
  if not(isempty(t))
    nixs = t(1,:);
    nepts = t(2,:); 
  end
else
  t = mlnet(mlix).back;
  if not(isempty(t))
    nixs = t(1,:);
    nepts = t(2,:);
  end
end

% filter out used main-lines connected at used endpts
t = find(not(used(nixs)));
nixs = nixs(t);
nepts = nepts(t);
tot_nixs = nixs;
tot_nepts = nepts;
tot_depths = ones(1,length(nixs))*cur_depth;

if depth_left == 1
  return;
end

% next depth (including branch if necessary)
used(nixs)=true;
for t = 1:length(nixs)
   [new_nixs new_nepts new_depths] = NextMLs(nixs(t), mlnet, 3-nepts(t), used, depth_left-1, cur_depth+1);
   % nicely swap direction so as to go towards the opposite endpoint of the next main-line
   % (opposite to the endpoint the current main-line was linked to)
   tot_nixs = [tot_nixs new_nixs];
   tot_nepts = [tot_nepts new_nepts];
   tot_depths = [tot_depths new_depths];
end

[trash b trash] = unique([tot_nixs; tot_nepts]', 'rows');  % allow for double connections (same ML, both endpts)
tot_nixs = tot_nixs(b);
tot_nepts = tot_nepts(b);
tot_depths = tot_depths(b);
