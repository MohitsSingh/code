function obj = PreprocessImage(img, do_all, model_img)

% Pre-processes image img, by computing
% all data used later at matching time.
% Only need edgelchains (in field img.ecs) as input.
% If do_all set, (re-)compute all data, if not
% compute only missing data.
%

if nargin < 2
  do_all = false;
end

obj = img;
if nargin < 3
  model_img = false;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute edgelchains data
if do_all || not(isfield(obj.ecs,'cv'))
  disp(['Fitting curves to edgelchains']);
  obj.ecs = FitCurve(obj.ecs);
end

if (do_all || not(isfield(obj,'diagrange'))) && model_img
  obj.diagrange = DiagRange(obj);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute data related to ML-net approach
if model_img
  obj.becs = BinEdgelChains(obj.ecs);
  obj.links = [];
elseif do_all || not(isfield(obj,'links'))
  disp('Finding links among edgel-chains (tangent-continuous)');
  obj.becs = BinEdgelChains(obj.ecs);
  obj.links = LinkEdgelChains(obj.ecs, obj.becs, true);   % true -> enforce tangent continuity
  t = FilterLinks(obj.ecs, obj.links, 30, 0.2);
  n = size(obj.links,2);
  obj.links = obj.links(:,setdiff(1:n,t));
  do_all = true;
end

if do_all || not(isfield(obj,'mainlines')) || not(isfield(obj,'eds'))
  disp(['Fitting regular contour segments']);
  [obj.mainlines obj.eds] = ConstructAllMainLines(obj.ecs);
  disp(['Fitting segments bridging over links']);
  [t_mls t_eds] = ConstructMainLinesAtLinks(obj.ecs, obj.links, size(obj.mainlines,2)+1);
  obj.mainlines = [obj.mainlines t_mls];
  obj.eds = [obj.eds; zeros(3,size(obj.eds,2))];
  obj.eds = [obj.eds t_eds];
  do_all = true;  % if MLs rebuilt, then need to rebuild everything
end

if (do_all || not(isfield(obj,'strengths'))) && not(model_img)
  disp(['Computing strengths of segments']);
  E = imread([obj.name '_edges.tif']);
  obj.strengths = ComputeMLStrengths(obj, double(E)/255);
  clear E;
end

if model_img
  obj.links_ntc = [];
elseif do_all | not(isfield(obj,'links_ntc'))
  disp(['Finding links among edgel-chains (no tangent-continuity constraint)']);
  obj.becs = BinEdgelChains(obj.ecs);
  obj.links_ntc = LinkEdgelChains(obj.ecs, obj.becs, false);
  t = FilterLinks(obj.ecs, obj.links_ntc, 30, 0.2);
  n = size(obj.links_ntc,2);
  obj.links_ntc = obj.links_ntc(:,setdiff(1:n,t));
  do_all = true;
end

if do_all | not(isfield(obj,'mlabc'))
  disp(['Computing implicit equation of straight contour segments'' support lines']);
  obj.mlabc = ComputeImplicitLines(obj.mainlines);
end

if do_all | not(isfield(obj,'S'))
  disp(['Computing all sidedness-orientation constraints']);
  obj.S = ComputeAllSidednessOrient(obj.mlabc);
end

if do_all | not(isfield(obj,'La'))
  disp(['Computing all location-angles and location-dists']);
  [obj.La obj.Ld] = ComputeAllLocAnglesDists(obj.mlabc);
end

if do_all | not(isfield(obj,'mlnet'))
  disp(['Building Contour Segment Network']);
  obj.mlnet = BuildMainLinesNetwork(obj.mainlines, obj.eds, obj.links_ntc, obj.ecs);
  do_all = true;
end

if do_all | not(isfield(obj,'nextmls3')) | not(isfield(obj,'epts'))
  disp(['Computing depth<=3 neighbors and linking pts']);
  [obj.nextmls3 obj.epts] = ComputeAllNextMLs(obj.mlnet, obj.mainlines, 3);
end
