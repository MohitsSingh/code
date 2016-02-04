function t = AddkASData(I, model, light, vak)

% Compute kas data
% if light -> avoid those expensive computations used only by the shape matcher det-ann approach
%%%%%%%%%%%%%%%%%%%

% if model given -> compute also kas-based edgel dissimilarities

if nargin < 2
  model = false;
end

if nargin < 3
  light = false;
end
 

disp(['Computing ' num2str(vak) 'AS data']);
I.bb = GetObjBB(I);
I.all_eds = GetAllEdgels(I, true);

I.pas = DetectkAS(I.mlnet,false,vak);
[I.pas I.pas_ls]= DescribekAS(I.pas, I.mainlines);
if not(isempty(I.all_eds))
  I.EP = ComputeEdgelsOnkAS_LUT(I, I.all_eds(2:3,:), I.bb);  % lookup table telling which edgels are covered by a pas
else
  I.EP = [];
end

fname = [I.name '_edges.tif'];
if exist(fname)
  I.ES = imread(fname);
else
  disp(['Warning: file ' fname ' not found. -> no edge-strengths loaded']);
end

% to be updated for kAS
% if not(light)   % sw-pas-svm doesn't need it (and it's so slow to compute!)
%   if not(islogical(model))
%     I.PASDall = AllEdgelsPASDissimilarity(model, I);
%   end
%   %
%   I.adj = AdjacentEdgels(I);
%   I.beds = BinEdgels(I.all_eds(2:3,:));
%   %I.SP = AllEdgelsSSSP(I, 30);  % 30: abs max path length ever considered
%   I.SP = AllEdgelsSSSP(I, 50);  % 50 -> hopefully linders various bad effects of current M -> matches procedure
% end



if isfield(I,'strengths')
  I.pas_strengths = ComputekASStrengths(I.pas, I.strengths);
else
  disp(['Warning: ' I.name ' has no strengths info. This is ok if ' I.name ' is hand-drawn.']);
end

if not(isfield(I, 'diagrange'))
  I.diagrange = DiagRange(I);
end


% keep all fields, including those needed only by CSN approach (backward compatibility)
t = I;
clear I;
return;  



% Keep only fields needed for the PAS-based object detection approach
% (useful to prune away other info needed only by the CSN approach)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t.ifname = I.ifname;
t.ecs = I.ecs;
t.becs = I.becs;
t.name = I.name;
t.mainlines = I.mainlines;
t.eds = I.eds;
t.mlnet = I.mlnet;
t.pas = I.pas;
t.pas_ls = I.pas_ls;
t.bb = I.bb;
t.all_eds = I.all_eds;
t.EP = I.EP;
t.links_ntc = I.links_ntc;        % needed for computing adjacent edgels (indeed can remove from here once development is over)
%
if isfield(I, 'ES')
  t.ES = I.ES;
end
%
if isfield(I,'strengths')
  t.strengths = I.strengths;
  t.pas_strengths = I.pas_strengths;
end
%
t.diagrange = I.diagrange;
%
if isfield(I, 'PASDall')
  %t.PASDall = I.PASDall;
end
%
t.adj = I.adj;
t.beds = I.beds;    
t.SP = I.SP;

clear I;
