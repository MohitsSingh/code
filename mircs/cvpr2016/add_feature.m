function f = add_feature(f,feats,name,abbr)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = length(f);
if n==0
    f = struct('feats',{},'name',{},'abbr',{});
end
n = n+1;
if nargin < 4
    abbr = name;
end
f(n).feats = feats;
f(n).name = name;
f(n).abbr = abbr;



% silly, but make sure that everyone has an abbreviation (backwards
% compatability)
for t = 1:n
    if isempty(f(t).abbr)
        f(t).abbr = f(t).name;
    end
end
end


