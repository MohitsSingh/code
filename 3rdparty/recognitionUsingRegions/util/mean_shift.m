function [drect,dscores,dprob]=mean_shift(rect,scores,param,mode)
    if nargin < 4,
        mode = 'max';
    end;
    indx = find(scores > param.th);
    if (length(indx) <=1)
        drect=rect(indx,:);
        dscores=scores(indx);
        dprob = ones(length(indx));
        return;
    end
    pthresh = 1/40;
    dthresh = 1e-2;
    minw = min(rect(indx,3));
    minh = min(rect(indx,4));
    
    %scale this to the size of the smallest
    param.sw = param.sw*minw;
    param.sh = param.sh*minh;
    
    p = [rect(indx,1)+0.5*rect(indx,3) rect(indx,2)+0.5*rect(indx,4) log(rect(indx,3)/minw) log(rect(indx,4)/minh)];
    w = scores(indx) - param.th;
    vars = [param.sw*exp(p(:,3)) param.sh*exp(p(:,4)) param.ss*ones(size(p,1),1) param.ss*ones(size(p,1),1)];
    vars = vars.^2;
    
    pmode = zeros(size(p));
    wmode = zeros(size(w));
    for i = 1:size(p,1)
        [pmode(i,:) wmode(i,:)] = compute_mode(i,p,w,vars,pthresh,mode);
    end
    [umode,uscore] = compute_unique_modes(pmode,wmode,dthresh);
    %convert to rectangles again
    sw = exp(umode(:,3))*minw;
    sh = exp(umode(:,4))*minh;
    drect = [umode(:,1)-0.5*sw umode(:,2)-0.5*sh sw sh];
    dscores = uscore + param.th;
    dprob = zeros(size(p,1),size(umode,1));
    for i = 1:size(umode,1)
        d = p - repmat(umode(i,:),size(p,1),1);
        dprob(:,i) = exp(-sum(d.^2./vars,2));
    end;
    dprob_sum = sum(dprob,2) + (sum(dprob,2)==0);
    dprob = dprob ./ repmat(dprob_sum,1,size(dprob,2));
end
%% compute modes
function [pmode,wmode]=compute_mode(i,p,w,vars,thresh,mode)
    pmode = p(i,:);
    wmode = w(i);
    npts = size(p,1);
    niters = 0;
    MAXITERS = 20000;
    while(1)
        d = p - repmat(pmode,[npts 1]);
        d = d.^2;
        wd = w.*exp(-sum(d./vars,2));
        wd = wd/sum(wd);
        pmode_new = wd'*p;
        if(mean(abs(pmode_new-pmode)) < 1e-5) || niters > MAXITERS,
            break;
        end
        pmode = pmode_new;
        if strcmp(mode,'max'),
            wmode = max(w(wd > thresh));
        else
            wmode = sum(w.*wd);
        end;
        
        niters = niters + 1;
    end
    if size(wmode,1) < 1,
        wmode = 0;
    end;
end
%% compute the unique modes
function [umode, uscore]=compute_unique_modes(pmode,wmode,thresh)
    npts = size(pmode,1);
    all=1:npts;
    uniq=[];
    
    while(~isempty(all))
        i=all(1);
        uniq = [uniq i];
        d = pmode(all,:) - repmat(pmode(i,:),size(all,2),1);
        d = mean(abs(d),2);
        samei=d<thresh;
        all(samei) = [];
    end
    umode = pmode(uniq,:);
    uscore = wmode(uniq,:);
end
