function res = milFeatures2(f_check,f_proto)


f_proto = cat(2,f_proto{:})';
res = zeros(size(f_proto,1),length(f_check));
for ii = 1:length(f_check)
    ii
    f = f_check{ii}';    
    aa = min(l2(f,f_proto))';
    res(:,ii) =aa;
end
