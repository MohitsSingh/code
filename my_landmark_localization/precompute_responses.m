function res = precompute_responses(param,patterns)
    ticID = ticStatus('precomputing responses...',.5,.5);
    n = length(patterns);
    res = cell(n,1);
    for t = 1:n
        curPattern = patterns{t};             
        r = img_detect(param,curPattern);        
        res{t} = {curPattern,r};
        tocStatus(ticID,t/length(param.patterns));
    end
    
    a = 0;
end