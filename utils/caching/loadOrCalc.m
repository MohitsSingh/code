function [ res ] = loadOrCalc(  fun_handle, initData, params, cachePath, dontReallyLoad)
%LOADORCALC Generic caching function, loads result from cachepath if it
%exists, or calculates it and saves if not; if cachePath is empty simply
% performs the calculation and returns.
if nargin < 5
    dontReallyLoad = false;
end
if (~isempty(cachePath))
    gotFile = false;
    if (exist(cachePath,'file'))
        if dontReallyLoad
            res = [];
            return
        end
        try
            res = load(cachePath); % assume that this contains 'res'
            gotFile = true;
        catch e
            disp(['deleting corrupt file: ' cachePath]);
            delete(cachePath);
        end
    end
    %         save(cachePath,'-struct','res');
    %L = load(U,'feats','moreData');
    if (~gotFile)
        res = feval(fun_handle,initData,params);
        try
            save(cachePath,'-struct','res');
        catch e
            
            save([cachePath '.error'],'params','e');
        end
    end
else
    res = feval(fun_handle,initData,params);
end
end