function [mls, eds] = FindMainLines(chain, cv, t, k, d, verbose)

% Find main lines of curve cv, with curvature k,
% discretised at points t (arclength).
% d are the tangent orientations.
%
% Criteria:
%   start from abs min curv pt, add pt-by-pt to segment until the max distance is bigger than some
%   threshold from the line going through the two endpoints, or if the difference in orientation
%   with the initial point exceeds another threshold
%   (use generous thresholds, as we want to find main lines, not real straight line segments)
%   At every iter, add point on the end causing the smallest mean distance.
%   Repeat from the next min curv pt, without using any edgel twice.
%

mls = [];
eds = [];

if nargin < 5
   verbose = false;
end

% is the chain closed ?
closed = ClosedChain(chain, 0.1, t(end));

% no edgel used up to now
used = zeros(1,length(t));

% initialize figure (for display in verbose mode)
if verbose
  figure, axis ij, hold on;
  %plot(chain(1,:), chain(2,:), 's', 'color', [0.7 0.8 0.7], 'LineWidth', 10);  % for paper figs
  plot(chain(1,:), chain(2,:), 'k', 'LineWidth', 1);
  %fnplt(cv,'k',3);  % for paper figs
  fnplt(cv,'g');
  xlabel('Main lines');
end

% prepare direction angles in [0,2pi]
a = atan2(d(2,:),d(1,:));
as0 = find(a<0);
a(as0) = a(as0)+2*pi;

% Find main lines
k([1 2 end-1 end]) = inf;   % do not want to start from an endpoint in any case
%fuck=30;
while not(all(used))        % iterates until some edgel has not been investigated

  % find abs min curv unused pt
  nusix = find(not(used));
  [trash tm] = min(abs(k(nusix)));
  tm = nusix(tm);
  used(tm)=true;
  ct = tm;                  % current points on main line
  if verbose
    plot(chain(1,tm), chain(2,tm), '*k', 'LineWidth', 3);
    disp(['new min curv pt']);
    keyboard;
  end

  % expand
  needmore = true;
  tf = tm;              % 'front' end of line
  tb = tm;              % 'back' end of line
  ctf=tf;               % cycled versions
  ctb=tb;
  while true
     % orientation difference in forward direction
     ectf=tf+1;       
     if ectf>length(t) & closed ectf=ectf-length(t); end
     if (ectf>length(t) & not(closed)) | used(ectf)
        ddf = inf;
     else
        ddf = AbsAngleDiff4Q(a(ectf),a(tm));
     end

     % largest edgel distance if add edgel in forward direction
     if (ectf>length(t) & not(closed)) | used(ectf)
        df = inf;
     else
        l = cross([chain(:,ctb); 1], [chain(:,ectf); 1])';
        l = l / sqrt(l(:,1).^2 + l(:,2).^2);
        %ll = sqrt(sum((chain(:,ctb)-chain(:,ectf)).^2));
        df = max(abs(l*[chain(:,ct); ones(1,length(ct))])); 
     end

     % orientation difference in backward direction
     ectb=tb-1;
     if ectb<1 & closed ectb=ectb+length(t); end
     if (ectb<1 & not(closed)) | used(ectb)
        ddb = inf;
     else
        ddb = AbsAngleDiff4Q(a(ectb),a(tm));
     end

     % largest edgel distance if add edgel in backward direction
     if (ectb<1 & not(closed)) | used(ectb)
        db = inf;
     else
        l = cross([chain(:,ectb); 1], [chain(:,ctf); 1])';
        l = l / sqrt(l(:,1).^2 + l(:,2).^2);
        %ll = sqrt(sum((chain(:,ectb)-chain(:,ctf)).^2));
        db = max(abs(l*[chain(:,ct); ones(1,length(ct))])); 
     end
     
     % ??
     %if ddb>pi/6 db=inf; end;
     %if ddf>pi/6 df=inf; end;

     % chose if and in which direction to expand
     if (db > 5 | ddb > pi/6) & (df > 5 | ddf > pi/6)
        break;
     else
        if df < db
           tf=tf+1;
           ctf=tf;
           if ctf>length(t) & closed ctf=ctf-length(t); end
           ct = [ct ctf];
           used(ctf)=true;
        else
           tb=tb-1;
           ctb=tb;
           if ctb<1 & closed ctb=ctb+length(t); end
           ct = [ct ctb];
           used(ctb)=true;
        end
     end
 
    if verbose & rem(length(ct),fuck)==0
      plot(chain(1,ctf), chain(2,ctf), 'or', 'LineWidth', 2);
      plot(chain(1,ctb), chain(2,ctb), '+g', 'LineWidth', 2);
      keyboard;
    end
  end  % expand current line

  % compute line length
  if ctf < ctb
    l = (t(end)-t(ctb))+t(ctf);                         % main line crosses end of chain
  else
    l = t(ctf)-t(ctb);
  end

  if l > 10                                             % keep only lines of length > 10
     orient = atan2(chain(2,ctf)-chain(2,ctb),chain(1,ctf)-chain(1,ctb));
     if orient<0 orient=orient+2*pi; end                % orientation in [0,2pi] (direction along the chain)
     ml = [mean(chain(:,[ctb ctf])')'; orient; l; tm];  % initial point tm is used below to sort main-lines along the chain
     mls = [mls ml];
     eds = [eds [ctb; ctf]];
     if verbose
       plot(chain(1,ctb), chain(2,ctb), 'or', 'LineWidth', 2);
       plot(chain(1,ctf), chain(2,ctf), '+b', 'LineWidth', 2);
       disp(['main line drawn']);
       keyboard;
     end
  end

end  % try to find another main line

% sort main-lines along the chain
if not(isempty(mls))
  [trash ixs] = sort(mls(5,:));
  mls = mls(:,ixs);
  eds = eds(:,ixs);
  mls = mls(1:4,:);                % drop initial point edgel index
end
