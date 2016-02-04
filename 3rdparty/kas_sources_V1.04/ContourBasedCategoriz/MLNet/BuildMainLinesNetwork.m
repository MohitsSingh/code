function mlnet = BuildMainLinesNetwork(mls, eds, links, ecs, verbose)

% Builds a main-lines network.
% Each main-line is connected at its back and front endpoints
% to a set of other main-lines, according to the following rules:
%
%  1. Two subsequent main-lines along the same edgelchain are connected
%     at the front of the first to the back of the second.
%
%  2. For a pair of edgelchains linked at the endpoints, the first/last main-line
%     of the first chain is connected to the first/last main-line of the second chain
%     (first or last depending on which endpoints are linked).
%     For a T-junction link, the first/last main-line of the first chain
%     is connected to the main-line of the second chain with the closest endpoint, and
%     to the main-line (along the same chain) connected to it at that endpoint (redundance to
%     improve robustness). In this fashion, the main-line of the first chain is connected to main-lines
%     of the second chain to the two sides of the linking point.
%     Moreover, if the linking point falls the middle of a main-line, instead of making a unstable decision,
%     we connect to both its endpoints.
%
%  3. A main-line constructed over the link between two edgelchains
%     is connected to the main-line of the 'front' edgelchain which comes next after
%     its front endpoint, and to the main-line of the 'back' edgelchain with comes previous before
%     its back endpoint.
%
%  4. Two bridging main-lines with consecutive endpts on the same edgelchain
%     (ie: no other entire ML inbetween and little overlap in edgels used) are connected.
%
%  5. If a bridging ML with no front/back connection because it covers an ec till the end, and
%     this ec is linked to another, then connect the bridging ML to the first/last ML in this other ec.
%
% The network also stores to which endpoint a main-line is connected.
%
% Output:
% mlnet(id).front(i) = [idi; ept] = main-line id is connected at its front endpoint to length(mln(id).front)
%                                   other main-lines. The ith one is idi and it is connected to its ept endpoint
% mlnet(id).back(i) = [idi; ept]  = same for back connections
% ept = {1,2}; 1 = back, 2 = front
%

if nargin < 5
  verbose = false;
end

if size(mls,2) == 0
  mlnet = [];
  return;
end

% Rule 1
%%%%%%%%%%%%%%%%%%%%%%%
if verbose
  disp(['Processing rule 1']);
  disp(['=================']);
end

% initialize output
mlnet1(size(mls,2)).front = [];
mlnet1(size(mls,2)).back  = [];

% initialize counters
curr_ec = mls(2,1);
first_mlix_of_curr_ec = 1;
mlixs_atlinks = find(mls(2,:)==-1);
if not(isempty(mlixs_atlinks))
  last_mlix = mlixs_atlinks(1)-1;  % last regular main-line (not bridging over a link)
else
  last_mlix = size(mls,2);
end

% build connections
for curr_mlix = 2:last_mlix
  curr_ml = mls(:,curr_mlix);
  if curr_ml(2) == curr_ec
    mlnet1(curr_mlix-1).front = [mlnet1(curr_mlix-1).front [curr_mlix; 1]];
    mlnet1(curr_mlix).back    = [mlnet1(curr_mlix).back [curr_mlix-1; 2]];
  else
    if ClosedChain(ecs(curr_ec).chain, 0.1, ecs(curr_ec).t(end))
      mlnet1(first_mlix_of_curr_ec).back = [mlnet1(first_mlix_of_curr_ec).back [curr_mlix-1; 2]];
      mlnet1(curr_mlix-1).front = [mlnet1(curr_mlix-1).front [first_mlix_of_curr_ec; 1]];
    end
    curr_ec = curr_ml(2);
    first_mlix_of_curr_ec = curr_mlix;
  end
end
if ClosedChain(ecs(curr_ec).chain, 0.1, ecs(curr_ec).t(end))
  mlnet1(first_mlix_of_curr_ec).back = [mlnet1(first_mlix_of_curr_ec).back [curr_mlix; 2]];
  mlnet1(curr_mlix).front = [mlnet1(curr_mlix).front [first_mlix_of_curr_ec; 1]];
end


% Rule 2
%%%%%%%%%%%%%%%%%%%%%%
%
if verbose
  disp(['Processing rule 2']);
  disp(['=================']);
end

% initialize output
mlnet2(size(mls,2)).front = [];
mlnet2(size(mls,2)).back  = [];

% lookup table reporting which main-lines lie on which edgelchain
mls_on_ec(length(ecs)).mlix = [];        % IMP: length(mls_en_cs) != length(ecs), as in the rare case where the last ec has no MLs -> crash later on !            
for mlix = 1:last_mlix
  ml = mls(:,mlix);
  mls_on_ec(ml(2)).mlix = [mls_on_ec(ml(2)).mlix ml(1)];
end

% do not consider twin links
valid = InvalidateTwinLinks(links);

% do not consider a link if there is not at least one main-line on each of the two linked edgelchains
for lix = 1:size(links,2)
  l = links(:,lix);
  if isempty(mls_on_ec(l(1)).mlix) | isempty(mls_on_ec(l(2)).mlix)
    valid(lix) = false;
  end
end

% build connections
for l = links(:,find(valid))

   % type of link
   if l(3)>0 & l(4)>0       % endpt-to-endpt
      if l(3) == 1 & l(4) == 1
         mlix1 = mls_on_ec(l(1)).mlix(1);
         mlix2 = mls_on_ec(l(2)).mlix(1);
         mlnet2(mlix1).back  = [mlnet2(mlix1).back  [mlix2; 1]];
         mlnet2(mlix2).back  = [mlnet2(mlix2).back  [mlix1; 1]];
      end
      if l(3) == 1 & l(4) == 2
         mlix1 = mls_on_ec(l(1)).mlix(1);
         mlix2 = mls_on_ec(l(2)).mlix(end);
         mlnet2(mlix1).back  = [mlnet2(mlix1).back  [mlix2; 2]];
         mlnet2(mlix2).front = [mlnet2(mlix2).front [mlix1; 1]];
      end
      if l(3) == 2 & l(4) == 2
         mlix1 = mls_on_ec(l(1)).mlix(end);
         mlix2 = mls_on_ec(l(2)).mlix(end);
         mlnet2(mlix1).front = [mlnet2(mlix1).front [mlix2; 2]];
         mlnet2(mlix2).front = [mlnet2(mlix2).front [mlix1; 2]];
      end
       if l(3) == 2 & l(4) == 1
         mlix1 = mls_on_ec(l(1)).mlix(end);
         mlix2 = mls_on_ec(l(2)).mlix(1);
         mlnet2(mlix1).front = [mlnet2(mlix1).front [mlix2; 1]];
         mlnet2(mlix2).back  = [mlnet2(mlix2).back  [mlix1; 2]];
      end
      % debug
      %if verbose
      %  disp(['Endpt-to-endpt link ' num2str(l')]);
      %	keyboard;
      % end
   else                  % T-junction
      if verbose
        disp(['T junction']);
        l
        DrawLinks(ecs, l);
      end

      % find out main-line of chain 2 closest to the linking point (l(5))
      [mlix2, ept] = NextMLOnChain(eds(:,mls_on_ec(l(2)).mlix), l(5));
      if EdInTheMiddle(l(5), eds(3:4,mlix2))  % is the linking point in the middle of mlix2 ? 
        epts = [1 2];  % If so, unstable decision about which endpt to connect to -> connect to both !
      else
        epts = ept;    % only connect to ept
      end
      omlixs = [];      % main-line on the other-side of the linking-pt than mlix2

      if l(3) == 1       % first endpt of first chain l(1)
        mlix1 = mls_on_ec(l(1)).mlix(1);
        for ept = epts
        mlnet2(mlix1).back = [mlnet2(mlix1).back [mlix2; ept]];             % add connection to closest ML ...
        if   ept == 1 mlnet2(mlix2).back   = [mlnet2(mlix2).back  [mlix1; 1]];
        else mlnet2(mlix2).front  = [mlnet2(mlix2).front [mlix1; 1]]; end
        %
        omlix = NML(mlnet1, mlix2, ept); oept = 3-ept; % ... and to the one on the other side of the linking point
        omlixs = [omlixs omlix];  % record in case of two other sides (when connecting to both endpts of mlix2)
        if not(isempty(omlix))
        mlnet2(mlix1).back = [mlnet2(mlix1).back [omlix; oept]];  
        if   oept == 1 mlnet2(omlix).back   = [mlnet2(omlix).back  [mlix1; 1]];
        else mlnet2(omlix).front  = [mlnet2(omlix).front [mlix1; 1]]; end
        end
        end % loop over epts   
      elseif l(3) == 2   % last endpt of first chain l(1)
        mlix1 = mls_on_ec(l(1)).mlix(end);
        for ept = epts
        mlnet2(mlix1).front = [mlnet2(mlix1).front [mlix2; ept]];             % add connection to closest ML ...
        if   ept == 1 mlnet2(mlix2).back   = [mlnet2(mlix2).back  [mlix1; 2]];
        else mlnet2(mlix2).front  = [mlnet2(mlix2).front [mlix1; 2]]; end
        %
        omlix = NML(mlnet1, mlix2, ept); oept = 3-ept; % ... and to the one on the other side of the linking point
        omlixs = [omlixs omlix];  % record in case of two other sides (when connecting to both endpts of mlix2)
        if not(isempty(omlix))
        mlnet2(mlix1).front = [mlnet2(mlix1).front [omlix; oept]];  
        if   oept == 1 mlnet2(omlix).back   = [mlnet2(omlix).back  [mlix1; 2]];
        else mlnet2(omlix).front  = [mlnet2(omlix).front [mlix1; 2]]; end
        end
        end % loop over epts
      end

      if verbose
        DrawMainLines(mls, [0 0 1], mlix1, 1, true, true);
        DrawMainLines(mls, [1 0 0], [mlix2 omlixs], 1, true, true);
        disp('T-junction link');
        disp(['From  ' num2str(mlix1) '  to  ' num2str([mlix2 omlixs])]); newline; newline;
        disp_mlnet(mlnet2, mlix1);
        disp_mlnet(mlnet2, mlix2);
        for omlix = omlixs disp_mlnet(mlnet2, omlix); end
        keyboard;
      end
   end  % type of link

end % loop over links



% Rule 3
%%%%%%%%%%%%%%%%%%%%%%
%

if verbose
  disp(['Processing rule 3']);
  disp(['=================']);
end

% initialize output
mlnet3(size(mls,2)).front = [];
mlnet3(size(mls,2)).back  = [];

% initialize counters
start_mlix = last_mlix+1;       % main-lines at links come right after conventional main-lines in mls
end_mlix = size(mls,2);
conn_count = 0;

% build connections
for curr_mlix = start_mlix:end_mlix

  curr_ml = mls(:,curr_mlix);
  ecb = eds(2,curr_mlix);  ecf = eds(5,curr_mlix); % front and back ec

  if verbose 
    DrawEdgelChains(ecs);
    DrawMainLines(mls(:,mls_on_ec(ecb).mlix), [1 0 0], 'all', 1, true);  % possible back connections
    DrawMainLines(mls(:,mls_on_ec(ecf).mlix), [0 1 0], 'all', 1, true);  % possible front connections    
    DrawMainLines(mls(:,curr_mlix), [0 0 1], 'all', 2);
    disp(['Processing bridging ML ' num2str(curr_mlix)]);
  end
  
  % back connection
  if not(isempty(mls_on_ec(ecb).mlix))
    %[mlix2 ept] = MainLineWithClosestEndpoint(mls(:,mls_on_ec(ecb).mlix), curr_ml, 1);
    [mlix2 ept] = NextMLOnChain(eds(:,mls_on_ec(ecb).mlix), eds([4 3],curr_mlix));  % 4 3 ->  reverse 'direction'
    if not(isempty(mlix2))
      mlnet3(curr_mlix).back = [mlnet3(curr_mlix).back [mlix2; ept]];
      if ept == 1
         mlnet3(mlix2).back   = [mlnet3(mlix2).back  [curr_mlix; 1]];
      else
         mlnet3(mlix2).front  = [mlnet3(mlix2).front [curr_mlix; 1]];
      end
      conn_count = conn_count + 1;
      if verbose
        disp(['Back connection']);
        DrawMainLines(mls(:,mlix2), [1 0 0], 'all', 2, true);
        disp(['Connected to ML ' num2str(mlix2)]);
        disp(['mlnet for bridging ML']);
        mlnet3(curr_mlix).front
        mlnet3(curr_mlix).back
        disp(['mlnet for connected ML']);
        mlnet3(mlix2).front
        mlnet3(mlix2).back
      end
    else % no connection
      if verbose
        disp(['No back connection']);
      end
    end
  end

  % front connection
  if not(isempty(mls_on_ec(ecf).mlix))
    %[mlix2 ept] = MainLineWithClosestEndpoint(mls(:,mls_on_ec(ecf).mlix), curr_ml, 2);
    [mlix2 ept] = NextMLOnChain(eds(:,mls_on_ec(ecf).mlix), eds(6:7,curr_mlix));
    if not(isempty(mlix2))
      mlnet3(curr_mlix).front = [mlnet3(curr_mlix).front [mlix2; ept]];
      if ept == 1
         mlnet3(mlix2).back   = [mlnet3(mlix2).back  [curr_mlix; 2]];
      else
         mlnet3(mlix2).front  = [mlnet3(mlix2).front [curr_mlix; 2]];
      end
      conn_count = conn_count + 1;
      if verbose
        disp('Front connection');
        DrawMainLines(mls(:,mlix2), [0 1 0], 'all', 2, true);
        disp(['Connected to ML ' num2str(mlix2)]);
        disp(['mlnet for bridging ML']);
        mlnet3(curr_mlix).front
        mlnet3(curr_mlix).back
        disp(['mlnet for connected ML']);
        mlnet3(mlix2).front
        mlnet3(mlix2).back
      end
    else % no connection
      if verbose
        disp(['No front connection']);
      end
    end
  end
  
end  % loop over main-lines at links
disp(['Rule 3 made ' num2str(conn_count) ' connections']);


% Rule 4
%%%%%%%%%%%%%%%%%%%%%%
%

if verbose
  disp(['Processing rule 4']);
  disp(['=================']);
end

% initialize output
mlnet4(size(mls,2)).front = [];
mlnet4(size(mls,2)).back  = [];

% initialize counters
start_mlix = last_mlix+1;       % main-lines at links come right after conventional main-lines in mls
end_mlix = size(mls,2);
conn_count = 0;

% lookup table reporting which bridging main-lines pass through which edgelchain ...
brid_mls_on_ec(length(ecs)).mlix = [];
% ... and which endpt is on the edgelchain (ed(1)) and which edgels it uses (ed(2:3))
brid_mls_on_ec(length(ecs)).ed = [];
for mlix = start_mlix:end_mlix
  brid_mls_on_ec(eds(2,mlix)).mlix = [brid_mls_on_ec(eds(2,mlix)).mlix eds(1,mlix)];
  brid_mls_on_ec(eds(2,mlix)).ed   = [brid_mls_on_ec(eds(2,mlix)).ed   [1; eds(3:4,mlix)]];
  brid_mls_on_ec(eds(5,mlix)).mlix = [brid_mls_on_ec(eds(5,mlix)).mlix eds(1,mlix)];
  brid_mls_on_ec(eds(5,mlix)).ed   = [brid_mls_on_ec(eds(5,mlix)).ed   [2; eds(6:7,mlix)]];
end

% process all edgelchains
for ecix = 1:length(brid_mls_on_ec)

  mlixs = brid_mls_on_ec(ecix).mlix;
  edt = brid_mls_on_ec(ecix).ed;   % temp ed (avoid name conflicts)
  
  if verbose & length(mlixs)>1
    DrawEdgelChains(ecs);
    DrawMainLines(mls(:,mls_on_ec(ecix).mlix), [0 0 1], 'all', 1, true);
    DrawMainLines(mls(:,mlixs), [0 1 0], 'all', 1, true);
    disp(['Processing edgelchain ' num2str(ecix)]);
  end
  
  for mlix_ix1 = 1:length(mlixs)
    mlix1 = mlixs(mlix_ix1);
    for mlix_ix2 = (mlix_ix1+1):length(mlixs)
      mlix2 = mlixs(mlix_ix2);
      overlap = IntervalsOverlap(sort(edt(2:3,mlix_ix1)), sort(edt(2:3,mlix_ix2)));
      inbetween = EntireMLsOnEdgels(eds(:,mls_on_ec(ecix).mlix), sort([edt(edt(1,mlix_ix1)+1,mlix_ix1) edt(edt(1,mlix_ix2)+1,mlix_ix2)]));
      if verbose    
        disp(['Should I connect ml ' num2str(mlixs(mlix_ix1)) ' to '  num2str(mlixs(mlix_ix2)) ' ?']);
	disp(['overlaps = ' num2str(overlap/(abs(edt(2,mlix_ix1)-edt(3,mlix_ix1))+1)) ', ' ...
	       num2str(overlap/(abs(edt(2,mlix_ix2)-edt(3,mlix_ix2))+1))]);
        disp(['MLs inbetween: ' num2str(inbetween)]);
      end
      if overlap/(abs(edt(2,mlix_ix1)-edt(3,mlix_ix1))+1) < 0.33 & overlap/(abs(edt(2,mlix_ix2)-edt(3,mlix_ix2))+1) < 0.33 & isempty(inbetween)
        % no ML is entirely inbetween the two bridging MLs and they don't overlap much -> connect
	if edt(1,mlix_ix1) == 2
          mlnet4(mlix1).front = [mlnet4(mlix1).front [mlix2; edt(1,mlix_ix2)]];
	else
	  mlnet4(mlix1).back  = [mlnet4(mlix1).back  [mlix2; edt(1,mlix_ix2)]];
	end
	if edt(1,mlix_ix2) == 2
          mlnet4(mlix2).front = [mlnet4(mlix2).front [mlix1; edt(1,mlix_ix1)]];
	else
	  mlnet4(mlix2).back  = [mlnet4(mlix2).back  [mlix1; edt(1,mlix_ix1)]];
	end
	conn_count = conn_count + 1;
     	if verbose
	  disp([ept2str(edt(1,mlix_ix1)) ' of bridging ML ' num2str(mlix1) ...
	        ' connected to ' ept2str(edt(1,mlix_ix2)) ' of bridging ML ' num2str(mlix2)]);
          disp(['mlnet for ml ' num2str(mlix1)]);
          mlnet4(mlix1).front
          mlnet4(mlix1).back
          disp(['mlnet for ml ' num2str(mlix2)]);
          mlnet4(mlix2).front
          mlnet4(mlix2).back
          keyboard;
	end
      end
    end
  end
end % loop over edgelchains
disp(['Rule 4 made ' num2str(conn_count) ' connections']);



% Rule 5
%%%%%%%%%%%%%%%%%%%%%%
%

if verbose
  disp(['Processing rule 5']);
  disp(['=================']);
end

% initialize output
mlnet5(size(mls,2)).front = [];
mlnet5(size(mls,2)).back  = [];

% initialize counters
start_mlix = last_mlix+1;       % main-lines at links come right after conventional main-lines in mls
end_mlix = size(mls,2);
conn_count = 0;

% build connections
%verbose = true;  % debug
for curr_mlix = start_mlix:end_mlix

   if verbose
     disp(['Processing bridging ML ' num2str(curr_mlix)]);
   end

   % does this bridging ML lack a connection ?
   if isempty(mlnet3(curr_mlix).back)
     % is the back ec linked to another ec ?
     ecix = eds(2,curr_mlix);
     t = find(valid);
     [curr_links links_ixs] = LinksForEC(ecix,ECDir(eds([4 3],curr_mlix),size(ecs(ecix).chain,2)),links(:,t));
     links_ixs = t(links_ixs);
     if not(isempty(curr_links))
       if verbose
         DrawLinks(ecs, links(:,links_ixs));
	 DrawMainLines(mls, [0 0 1], curr_mlix, 2, true);  % current bridging ML
	 disp(['Bridging ML ' num2str(curr_mlix) ' lacks back connection. Displaying next links.']);
       end
       for l = curr_links
         if verbose disp(['back of bridging ML ' num2str(curr_mlix) ' connected to ']); end
         if l(2) == 1
	   conn_mlix = mls_on_ec(l(1)).mlix(1);
           mlnet5(curr_mlix).back   = [mlnet5(curr_mlix).back   [conn_mlix; 1]];
           mlnet5(conn_mlix).back   = [mlnet5(conn_mlix).back   [curr_mlix; 1]];
	   if verbose
	     disp(['back of ML ' num2str(conn_mlix)]);
	     DrawMainLines(mls, [1 0 0], conn_mlix, 1, true);
	     keyboard;
	   end
	   conn_count = conn_count + 1;
	 elseif l(2) == 2
	   conn_mlix = mls_on_ec(l(1)).mlix(end);
           mlnet5(curr_mlix).back   = [mlnet5(curr_mlix).back   [conn_mlix; 2]];
           mlnet5(conn_mlix).front  = [mlnet5(conn_mlix).front  [curr_mlix; 1]];
	   if verbose
	     disp(['front of ML ' num2str(conn_mlix)]);
	     DrawMainLines(mls, [1 0 0], conn_mlix, 1, true);
	     keyboard;
	   end
	   conn_count = conn_count + 1;
	 elseif l(2) == 0  % T-junction
	   if verbose
	     disp(['T-junction: add no connection (avoid too many clutter connections)']);
	   end
	   % do not add connection (avoid too many clutter connections)
	 end
       end % loop over curr_links
     end
   end
   
   if isempty(mlnet3(curr_mlix).front)
     % is the front ec linked to another ec ?
     ecix = eds(5,curr_mlix);
     t = find(valid);
     [curr_links links_ixs] = LinksForEC(ecix,ECDir(eds([6 7],curr_mlix),size(ecs(ecix).chain,2)),links(:,t));
     links_ixs = t(links_ixs);
     if not(isempty(curr_links))
       if verbose
         DrawLinks(ecs, links(:,links_ixs));
	 DrawMainLines(mls, [0 0 1], curr_mlix, 2, true);  % current bridging ML
	 disp(['Bridging ML ' num2str(curr_mlix) ' lacks front connection. Displaying next links.']);
       end
       for l = curr_links
         if verbose disp(['front of bridging ML ' num2str(curr_mlix) ' connected to ']); end
         if l(2) == 1
	   conn_mlix = mls_on_ec(l(1)).mlix(1);
           mlnet5(curr_mlix).front  = [mlnet5(curr_mlix).front  [conn_mlix; 1]];
           mlnet5(conn_mlix).back   = [mlnet5(conn_mlix).back   [curr_mlix; 2]];
	   if verbose
	     disp(['back of ML ' num2str(conn_mlix)]);
	     DrawMainLines(mls, [1 0 0], conn_mlix, 1, true);
	     keyboard;
	   end
	   conn_count = conn_count + 1;
	 elseif l(2) == 2
	   conn_mlix = mls_on_ec(l(1)).mlix(end);
           mlnet5(curr_mlix).front  = [mlnet5(curr_mlix).front  [conn_mlix; 2]];
           mlnet5(conn_mlix).front  = [mlnet5(conn_mlix).front  [curr_mlix; 2]];
	   if verbose
	     disp(['front of ML ' num2str(conn_mlix)]);
	     DrawMainLines(mls, [1 0 0], conn_mlix, 1, true);
	     keyboard;
	   end
	   conn_count = conn_count + 1;
	 elseif l(2) == 0  % T-junction
	   if verbose
	     disp(['T-junction: add no connection (avoid too many clutter connections)']);
	   end
	   % do not add connection (avoid too many clutter connections)
	 end
       end % loop over curr_links
     end
   end
   
end % loop over bridging MLs
disp(['Rule 5 made ' num2str(conn_count) ' connections']);


% Fuse together all connections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mlnet = FuseMLNets(mlnet1, mlnet2);
mlnet = FuseMLNets(mlnet,  mlnet3);
mlnet = FuseMLNets(mlnet,  mlnet4);
mlnet = FuseMLNets(mlnet,  mlnet5);
