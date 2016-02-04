% weakly supervised object detection.

initpath;
config;
rmpath /home/amirro/code/3rdparty/exemplarsvm/VOCcode/
%% get image names, ground-truth labels

VOCinit;
ids_train = textread(sprintf(VOCopts.imgsetpath,VOCopts.trainset),'%s');
ids_val = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s');

classes_train = zeros(length(ids_train),length(VOCopts.classes));
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    [ids,curGT] = textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.trainset),'%s %d');
    classes_train(:,i) = curGT;
end
classes_val = zeros(length(ids_val),length(VOCopts.classes));
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    [ids,curGT] = textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');
    classes_val(:,i) = curGT;
end
%% extract deep features from all images.
train_feats = {};
featureExtractor = DeepFeatureExtractor(conf);
train_paths = {};
for t = 1:length(ids_train)
    train_paths{t} = sprintf(VOCopts.imgpath,ids_train{t});
end
val_paths = {};
for t = 1:length(ids_val)
    val_paths{t} = sprintf(VOCopts.imgpath,ids_val{t});
end

net = init_nn_network('imagenet-vgg-s.mat');
train_feats = extractDNNFeats(train_paths,net,16,false, false);
val_feats = extractDNNFeats(val_paths,net,16,false, false);
save ~/storage/misc/pas_feats.mat train_feats val_feats
%% train classifiers
ws = {};
x = train_feats.x;
x_val = val_feats.x;
res = struct;
net.layers = net.layers(1:15);
for iClass = 1:length(VOCopts.nclasses)
    iClass= 8
    lambdas =  [1e-5 1e-6 1e-7];% 1e-6]
    p = Pegasos(x,2*(classes_train(:,iClass)==1)-1,'lambda', lambdas);
%     ws{iClass} = p.w;
    res(iClass).w = p.w;
    curLabels = 2*(classes_val(:,iClass)==1)-1;    
    curScores =  p.w(1:end-1)'*x_val;    
    res(iClass).curScores = curScores;    
    [res(iClass).recall, res(iClass).precision, res(iClass).info] = vl_pr(curLabels,curScores);
    vl_pr(curLabels,curScores)
    figure(1)
    [z,iz] = sort(curScores,'descend');
    displayImageSeries(conf,val_paths(iz(1:100:end)),.1)
    
    %%
    figure(1);
    clf;
    for u = iz(1:10:end)
        [I,R] = getRegionImportance(conf,val_paths(u), res(iClass).w, net,[15]);
        clf; imagesc2(R);
        dpc
    end
    %%
end


