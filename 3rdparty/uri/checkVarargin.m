function [out varargout] = checkVarargin(varargin, field, defaultValue)
% The function determines whether optional inputs to a function contain a
% particular (field,value) pair. The function searches for the given field
% and returns the appropriate value. If the field is not passed then the 
% function returns a default value.

% varargin      - optional parameters
% field         - string denoting variable name
% defaultValue  - default value for use in case the field was not supplied


if (nargin < 2)
    error('at least one element must be passed as argument');
end

% if (nargin < 4)
%     removeFromVarargin = false;
% end

idxStrArg = cellfun(@isstr, varargin);
strArgs = varargin(idxStrArg);
idxStrArg = find(idxStrArg);

fieldIdx = strmatch(field, strArgs, 'exact');
if (fieldIdx)
    out = varargin{idxStrArg(fieldIdx)+1};
else
    if (nargin > 2)
        out = defaultValue;
    else
        out = [];
    end
end

if (nargout > 1) 
    varargout{1} = setdiff(1:length(varargin), [idxStrArg(fieldIdx) idxStrArg(fieldIdx)+1]);
end
