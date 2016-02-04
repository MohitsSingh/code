function Aucm = ucmStrength(A,L,ucm)
% Aucm = ucmStrength(L,ucm) find the mean strength of UCM between
% regions
[ii,jj] = find(A);
Aucm = inf(size(A));
dilatedRegions = {};
for k = 1:size(A,1)
    dilatedRegions{k} = imdilate(L==k)
end
    
for v1 = ii
    a1 = imdilate(L==v1);
    for v2 = jj
        a2 = imdilate(L==v2)
        if (jj < ii)
            continue;
                % slightly dilate both 
        end
    end
end
end