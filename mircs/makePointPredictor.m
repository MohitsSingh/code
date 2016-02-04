function res = makePointPredictor(feats,pts,n,s)
if (nargin < 3)
    n = 1000;
end
if nargin < 4
    s = 3;
end
res = struct('ferns_x',{},'ferns_y',{});
[res(1).ferns_x] = fernsRegTrain(double(feats'),pts(:,1),'loss','L2','eta',1,...
    'thrr',[-1 1],'reg',0.1,'S',s,'M',n,'R',3,'verbose',1);
[res(1).ferns_y] = fernsRegTrain(double(feats'),pts(:,2),'loss','L2','eta',1,...
    'thrr',[-1 1],'reg',0.1,'S',s,'M',n,'R',3,'verbose',1);
