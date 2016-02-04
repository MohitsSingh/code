function [mainlines, eds] = ConstructAllMainLines(ecs)

% Constructs main-lines for all chains ecs(i).chain
% (see notes 23.11.04)
% Needs curves ecs(i).cv, to be built beforehand by FitCurve
%
% Output:
% mls(:,i) are 6x1 vectors in the form:
%
% [id chain x y orientation length]'
%
% with orientation in [0,2pi]
%

mainlines = [];
eds = [];
id=1;
for ec_ix = 1:length(ecs)
  [mls edst] = ConstructMainLines(ecs(ec_ix), false);      % false -> no verbose
  if not(isempty(mls))
    mls = [ones(1,size(mls,2))*ec_ix; mls];                % prepend chain id
    mls = [id:(id+size(mls,2)-1); mls];                    % prepend main-line id
    mainlines = [mainlines mls];
    
    edst = [ones(1,size(mls,2))*ec_ix; edst];              % prepend chain id
    edst = [id:(id+size(mls,2)-1); edst];                  % prepend main-line id
    eds = [eds edst];
    
    id = id+size(mls,2);
  end
end
