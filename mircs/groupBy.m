function [res,H] = groupBy(s,fName)
    % set up a "hash-table".    
    H = hash;
    t = 1;
    r = zeros(1,length(s));   
    for k = 1:length(s)
        v = eval(sprintf('s(%d).%s',k,fName));
        if (isempty(H(v)))
            H(v) = t;
            r(k) = t;
            t = t+1;
        else
            m = H(v);
            r(k) = m;
        end
    end
    u = unique(r);
    for k = 1:length(u)
        res(k).group = s(r==u(k));
        res(k).key =eval(sprintf('res(%d).group(1).%s',k,fName));
    end