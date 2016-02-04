function fig_hs = DrawkASBG(obj, col, ids, thickness, draw_id, single_fig, best_svm)
  vak=Value_of_vak(size(obj.pas,1));

% Draws kAS obj.pas on new figures
% with edgelchains obj.ecs as background
%
% Input:
% obj.pas(:,i) are vakx1 vectors in the form [mlid1 mlid2 ... mlidvak]'
%
% draw_id    == true --> draws the segments ids (not the pas ids !)
% single_fig == true --> draws all requested pas on a single figure, even if some overlap
% single_fig == 2    --> draw all pas on the current figure (don't open a new one)
%
% ids = either a list (1-dim) -> draw only pas involving at least one segment of ids,
%           or a 2xN matrix   -> draw pas ids(:,i) (remember that pas(1,i) < pas(2,i) !)
%           or 'all'          -> draw all obj.pas
%
% Return handles to all figures it opened.
%

% process arguments
if nargin < 4
  thickness = 1;
end

if nargin < 5
  draw_id = false;
end

if nargin < 6
  single_fig = false;
end

if nargin < 7
  best_svm = false;
end

% shortcuts
mls = obj.mainlines;

% which pas to draw ?
if ischar(ids)
  pas_ids = 1:size(obj.pas,2);                     % if ids == 'all'
  tobedone_pas = ones(1,size(obj.pas,2));
  ids = obj.pas(1:vak,:);
else
  tobedone_pas = zeros(1,size(obj.pas,2));
  if size(ids,1) == 1  & not(best_svm)                            % ids is a list of segments ids
    for i=1:vak
      tobedone_pas(ismember(obj.pas(i,:),ids)) = true;% pas involving at least one segment of ids
    end;
    ids = obj.pas(1:vak,find(tobedone_pas));
  elseif size(ids,1) == vak % ids is a list of pas
    [ids b c] = unique(ids','rows');               % remove potential duplicates, and SORTS ids
    ids = ids';
    ids = ids(:,c);                                % remove sorting -> good to keep coloring consistent !
    tobedone_pas(ismember(obj.pas(1:vak,:)', ids', 'rows')') = true;
    if size(ids,2) > sum(tobedone_pas)
      disp('DrawkASBG: Warning: some requested pas ids do not exist.');
    end
  elseif size(ids,1) > vak
    error('DrawkASBG: ids must have at most 2 rows');
  end
end

%tobedone_pas = rand(1,length(tobedone_pas));  % for paper figure
%tobedone_pas = tobedone_pas > 0.5;            % only draw some pas
%keyboard;

% draw all pas to be done
fig_hs = [];
if not(any(tobedone_pas))
  DrawEdgelChains(obj.ecs, not(single_fig==2), false, false, true); % draw b/w ecs
  fig_hs = gcf;
end
while any(tobedone_pas)
  % Draws background (= edgelchains) in new figure
  DrawEdgelChains(obj.ecs, not(single_fig==2), false, false, true); % draw b/w ecs
  
  fig_hs = [fig_hs gcf];

  % Draws PAS (if not single_fig -> with non-overlap guarantee)
  hold on;
  col_id = 0;
  used_mls = zeros(1,size(obj.mainlines,2));
  for p = ids
    bool=(obj.pas(1,:)==p(1));
    for i=2:vak
        bool=bool & (obj.pas(i,:)==p(i));
    end
    pix = find(bool); 

    if tobedone_pas(pix)
      if not(any(used_mls(p))) | single_fig
        col_id = col_id+1;
        DrawEdgelsUnderMLs(obj, p(1:vak)', col_id, thickness, draw_id);
        used_mls(p(1:vak)) = true;
        tobedone_pas(pix) = false;
      end % still free mls
    end % pas to done ?
  end % loop over all pas
  if any(tobedone_pas) disp(['Type return for next subset of kas']); pause; end
      %keyboard; end
end % loop over figures (so that no two pas overlap)
