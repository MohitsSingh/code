initpath;
config;

conf.pasClass = 'bottle';
% conf.datasetManager = datasetManager('pascal');

bottleclass = 5;
cls = bottleclass;
[train_ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_train']),'%s %d');
train_labels = t==1;
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);
conf.suffix = 'pascal_bottle';
% on purpose, so there aren't too many patches
conf.pasMode = 'train';
clustering3(conf,discovery_sets,natural_set,'clusteringConf',conf,'ovp',.1);

% natural_set = {ids(1:2:end),ids(2:2:end)};

%% testing...
[val_ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_val']),'%s %d');
conf.pasMode = 'test';

load ~/storage/data/cache/detectors_5pascal_bottle

clusters = makeLight(clusters,'sv','vis','cluster_samples');
save bottle_clusters_lite clusters
toSave =1;

% val_ids = val_ids(t==1);
% val_ids = val_ids(1:10);
% t = t(t==1);
% t = t(1:10);
matlabpool
[dets] = getDetections(conf,val_ids,clusters,[],[],0)
[topdets] = getTopDetections(conf,dets,clusters,true,inf);
[vis,allImgs] = visualizeClusters(conf,val_ids,topdets);
imwrite(clusters2Images(vis),'bottle1.jpg');
[prec,rec,aps,T,M] = calc_aps(topdets,t==1);
[p,ip] = sort(aps,'descend');

ks  = 1:1:50
nq =15;
ps = zeros(nq,length(ks))
for q = 1:nq
    for ik = 1:length(ks)
        ik
        [w b ap] = checkSVM( M(:,ip(1:ks(ik))),t==1);
        ps(q,ik) = ap;
    end
end
plot(mean(ps,1))

qq = 13;
[w b ap] = checkSVM( M(:,ip(1:qq)),t==1);

[vis,allImgs] = visualizeClusters(conf,val_ids,topdets(ip(1:10)),...
    'disp_model',true,'nDetsPerCluster',20,'add_border',true,'gt_labels',t==1);

imwrite(clusters2Images(vis),'bottle_test.jpg');


X = M;
X(isinf(X(:))) = min(X(~isinf(X(:))));
s = X(:,ip(1:qq))*w;

[s,is] = sort(s,'descend');
plot(is)

% show the detections!!
p_sel = ip(1:qq);
for k = 1:length(is)
    locCount = 0;
    curLocs = zeros(length(p_sel),12);
    for q = 1:length(p_sel)
        locs_ = topdets(p_sel(q)).cluster_locs;
        r = find(locs_(:,11) == is(k),1,'first');
        if (~isempty(r))
            locCount= locCount+1;
            curLocs(locCount,:) = locs_(r,:);
        end
    end
    curLocs = curLocs(1:locCount,:);
    I = getImage(conf,val_ids{is(k)});
    figure(1);clf;
    imshow(I);
    hold on;
    plotBoxes2(curLocs(:,[2 1 4 3]),'g','LineWidth',2);
    pause;
end


