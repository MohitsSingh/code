function m = makeIDMap(d)
m = containers.Map('KeyType','int32','ValueType','any');
for t = 1:length(d)
    m(d(t).id) = d(t);
end
end