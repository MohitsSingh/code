function displayImageSeries(conf,ims,delay,trues,displayTrue,indexToShow)
if (nargin < 3)
    delay = 0;
end

if (nargin < 4)
    trues = true(size(ims));
end

if (nargin < 5)
    displayTrue = 0;
end

if (nargin < 6)
    indexToShow = 1:length(ims);
end

if (isstruct(ims))
    ims2 = ims;
    ims = {};
    for t = 1:length(ims2)
        ims{t} = ims2(t);
    end
end

if (isstruct(conf))
    for k = 1:length(ims)
        fprintf('%d, %3.0f%%\n',k,double(100*sum(trues(1:k)))/sum(trues));
        I = ims{k};
        if (isstruct(I) || ischar(I)), I = getImage(conf,I); end;
        
        if (displayTrue==1 && ~trues(k))
            continue
        end
        if (displayTrue==-1 && trues(k))
            continue
        end
        
    
        
        clf; imagesc2(I);
        title(num2str(indexToShow(k)));
        if (delay == 0)
            pause
        elseif (delay > 0)
            pause(delay);
        end
        
        drawnow
    end
end