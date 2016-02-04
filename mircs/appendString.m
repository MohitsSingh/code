function [A] = appendString(A,m,n,s,debug_)
%APPENDSTRING Summary of this function goes here
%   Detailed explanation goes here

if (nargin < 5)
    debug_ = false;
end
% debug_ = false;%TODO!!! notice this 
if (~debug_) 
    return;
end
if (~isempty(n))
    A{m,n}{end+1} = s;
else
    [ii,jj] = find(m);
    for k = 1:length(ii)
        A{ii(k),jj(k)}{end+1} = s;
    end
end
end