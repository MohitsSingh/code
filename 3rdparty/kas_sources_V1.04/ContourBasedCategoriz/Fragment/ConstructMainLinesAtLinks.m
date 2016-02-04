function [tot_mls, eds] = ConstructMainLinesAtLinks(ecs, links, first_id, verbose)

% Constucts main-lines on edgel chains ecs which pass through the links.
% Hence, each returned main-line covers parts of both edgelchains involved in a link.
% If two links link the same two edgelchains at the same endpoints, only the first in
% the list 'links' is considered (to avoid constructing pairs of nearly identical main-lines).
%
% Beware: - could be meaningful/desired to extend to follow links
%           'recursively' and produce main-lines bridging >2 edgelchains
%      
%
% Input:
% see DrawLinks and LinkEdgelChains for format of links
%
% Output:
% eds(:,id) = [id; back_ec; first_edgel; last_edgel; front_ec; first_edgel; last_egdel]
% useful for later building main-lines network
%

global colors;           % debug

if nargin < 4
  verbose = false;
end

% Invalidate twin links
valid = InvalidateTwinLinks(links);

tot_mls = [];
eds = [];
for l = links(:,find(valid))

  % Current link l has form
  % l = [chain1; chain2; endpt1; endpt2; link_pt]
  link_pt = l(5);

  % The first part of the joint chain is chain1
  chain1 = ecs(l(1)).chain;
  length1 = size(chain1,2);
  if l(3) == 1                                       % if endpoint is the start ...
     chain1 = reverse([chain1(1,:); chain1(2,:)]);   % ... reverse the chain
     lastpt1 = 1;
     dir1 = -1;
  else
     lastpt1 = size(chain1,2);
     dir1 = 1;
  end

  % The second part of the joint chain is chain2, starting from the linking point,
  % going 'away' from the endpoint of chain1 (endpt1)
  chain2 = ecs(l(2)).chain;
  length2 = size(chain2,2);
  if link_pt == size(chain2,2)
     dir2 = -1;
     lastpt2 = 1;
  elseif sqrt(sum((chain2(:,link_pt)-chain1(:,end)).^2)) < sqrt(sum((chain2(:,link_pt+1)-chain1(:,end)).^2))
     dir2 = 1;     % direction as in the chain order
     lastpt2 = size(chain2,2);
  else
     dir2 = -1;    % opposite direction
     lastpt2 = 1;
  end
  chain2 = chain2(:,link_pt:dir2:lastpt2);
  
  % Join the chains
  joint_chain = [chain1 chain2];
  
  % Fit curve and construct fragments on the joint chain
  joint_ec.chain = joint_chain;
  joint_ec = FitCurve(joint_ec);
  [mls ml_eds] = ConstructMainLines(joint_ec);
 
  % Keep only main-lines bridging the two edgelchains
  % that is: covering edgels from both chains (use info in ml_eds)
  % should be at most one
  bridging_mls = [];
  for mlix = 1:size(mls,2)
     if ml_eds(1,mlix) <= size(chain1,2) & ml_eds(2,mlix) > size(chain1,2)
        bridging_mls = [bridging_mls mlix];              
     end
  end
  if length(bridging_mls) > 1
    disp(['ConstructMainLinesAtLinks. Warning: more than 1 bridging main-lines at link ' num2str(l')]);
  end
  mls = mls(:,bridging_mls);

  if not(isempty(mls))
    mls = [-1*ones(1,size(mls,2)); mls];                % prepend -1 as chain id, as all constructed main-lines lie inbetween two chains
    mls = [first_id:(first_id+size(mls,2)-1); mls];     % prepend main-lines ids
    if dir1 == 1
       first_eds1 = ml_eds(1,bridging_mls);
    else
       first_eds1 = length1-ml_eds(1,bridging_mls)+1;
    end
    if dir2 == 1
       last_eds2 = ml_eds(2,bridging_mls)-length1+link_pt-1;
    else
       last_eds2 = link_pt-(ml_eds(2,bridging_mls)-length1)+1;
    end
    eds = [eds ...
    [first_id:(first_id+size(mls,2)-1); l(1)*ones(1,size(mls,2)); first_eds1; lastpt1*ones(1,size(mls,2));...
     l(2)*ones(1,size(mls,2)); link_pt*ones(1,size(mls,2)); last_eds2]];
    first_id = first_id+size(mls,2);
    tot_mls = [tot_mls mls];
    if verbose
      DrawLinks(ecs, l);
      fnplt(joint_ec.cv,'g'); 
      DrawMainLines(mls, [0.3 0 1]);
      keyboard;
    end
  end

end
