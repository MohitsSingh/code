if (~exist('initialized','var'))
    cd ~/code/mircs
    initpath;
    config;
    addpath('~/code/3rdparty');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');
    load('~/storage/misc/images_and_face_obj_full_imdb.mat')
    seg_dir = '~/storage/fra_db_seg_full';
    isTrain = find([fra_db.isTrain]);
    initialized = true;
    %     addpath('~/code/3rdparty/matconvnet-1.0-beta16/matlab');
    featureExtractor = DeepFeatureExtractor(conf,true,33);
    
     train_params = struct('classes',1:5,'toBalance',0,'lambdas',.001);
    train_params.task = 'classification';
    train_params.minGroupSize = 0;
    train_params.maxGroupSize = inf;
    train_params.hardnegative = false;
    train_params.standardize=false;
    train_params.minGroupSize=1;
    train_params.maxGroupSize=inf;
    train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
    
    nClasses=5;
end
%
% load the segmentations...
%%
%baseLinePath = '~/storage/misc/baseline_alexnet';
baseLinePath = '~/storage/misc/baseline_vgg16';
if exist(baseLinePath,'file')
    load(baseLinePath);
else    
    imgs_local = {};
    imgs_local_oracle = {};
    for t = 1:length(fra_db)
        I = imdb.images_data{t};
        if fra_db(t).isTrain
            faceBox = fra_db(t).faceBox;
        else
            faceBox = fra_db(t).faceBox_raw;
        end
        faceBox = inflatebbox(faceBox,2,'both',false);
        faceBox = round(faceBox);
        I_sub = cropper(I,faceBox);
        imgs_local{t} = I_sub;
    end
    
    % extract masks/bounding boxes around the action objects.
    objects_masked = {};
    valids = false(size(fra_db));
    for t = 1:length(fra_db)
        t
        I = imdb.images_data{t};
        cur_obj_mask = imdb.labels{t}>=3;
        if none(cur_obj_mask(:))
            objects_masked{t} = zeros(50,50,3,'uint8');
            continue
        end
        objects_masked{t} = maskedPatch(I,cur_obj_mask,true);
    end
        
    objects_masked_tight = {};
    for t = 1:length(fra_db)
        t
        I = imdb.images_data{t};
        cur_obj_mask = imdb.labels{t}>=3;
        if none(cur_obj_mask(:))
            objects_masked_tight{t} = zeros(50,50,3,'uint8');
            continue
        end
            valids(t) = true;
        objects_masked_tight{t} = maskedPatch(I,cur_obj_mask,true);
    end
    
    objects_boxed = {};
    for t = 1:length(fra_db)
        t
        I = imdb.images_data{t};
        cur_obj_mask = imdb.labels{t}>=3;
        if none(cur_obj_mask(:))
            objects_boxed{t} = 128*ones(50,50,3,'uint8');
            continue
        end
        bb = region2Box(cur_obj_mask);
        bb(1:2) = bb(1:2)-1;
        bb(3:4) = bb(3:4)+1;
        objects_boxed{t} = cropper(I,round(bb));
    end
    
    feats_objects_masked = featureExtractor.extractFeaturesMulti(objects_masked,false);
    feats_objects_masked_tight = featureExtractor.extractFeaturesMulti(objects_masked_tight,false);
    feats_objects_boxed = featureExtractor.extractFeaturesMulti(objects_boxed,false);
    feats_local = featureExtractor.extractFeaturesMulti(imgs_local,false);
    feats_global = featureExtractor.extractFeaturesMulti(imdb.images_data,false);
    labels = [fra_db.classID];
    isTrain = [fra_db.isTrain]
   
    train_params = struct('classes',1:5,'toBalance',0,'lambdas',.001);
    train_params.toBalance =0;
    train_params.task = 'classification';
    train_params.hardnegative = false;
    nClasses = 5;
    train_params.classes = 1:nClasses;
    all_feats = struct('feats',{},'name',{});
    all_feats(1).feats = feats_global;
    all_feats(1).name = 'global features';
    all_feats(1).abbr = 'G';
    all_feats(2).feats = feats_local;
    all_feats(2).name = 'near face features';
    all_feats(2).abbr = 'F';
    all_feats(3).feats = feats_objects_masked;
    all_feats(3).name = 'objects_masked';
    all_feats(3).abbr = 'M';
    all_feats(4).feats = feats_objects_masked_tight;
    all_feats(4).name = 'objects_masked_tight';
    all_feats(4).abbr = 'MT';
    all_feats(5).feats = feats_objects_boxed;
    all_feats(5).name = 'objects_boxed';
    all_feats(5).abbr = 'BB';
        
    
    
    
    res = train_and_test_helper(all_feats,labels,isTrain,valids,train_params);
    % res = apply_classifiers(res,feats_global(:,~sel_train),labels(~sel_train),train_params);   
    save(baseLinePath,'res','all_feats','train_params','objects_boxed','objects_masked','objects_masked_tight',...
        'imgs_local');
end

%%
% summarize results
[summary,sm_baseline] = summarizeResults(res,all_feats,train_params)
matrix2latex(table2array(sm_baseline), 'figures/baseline.tex','format','%0.3f','rowLabels',sm_baseline.Properties.RowNames,'columnLabels',sm_baseline.Properties.VariableNames);
% matrix2lyx(summary, '1.lyx', '%02.0f');

%% find the distribution of face box size vs image size
showSomeResults(res,outPath,fra_db,imdb,16,1)

%% 
% extract features from all of s40-fra
% load ~/storage/mircs_18_11_2014/s40_fra.mat

s40_fra_classes = [s40_fra.classID];
isTrain = [s40_fra.isTrain];
conf.get_full_image = true;
imgs=  {};
for t = 1:length(s40_fra)
    t
%     if ~isempty(imgs{t})
%         imgs{t} = im2uint8(imgs{t});
%     else
        imgs{t} = im2uint8(getImage(conf,s40_fra(t)));
%     end
end
    s
all_feats_fc6 = featureExtractor.extractFeaturesMulti(imgs);



%%
train_params = struct('classes',1:40,'toBalance',0,'lambdas',.001);
train_params.task = 'classification';
train_params.minGroupSize = 0;
train_params.maxGroupSize = inf;
train_params.hardnegative = false;
train_params.standardize=false;
train_params.minGroupSize=1;
train_params.maxGroupSize=inf;
train_params.classNames = row(conf.classes);

s40_feats = add_feature([],all_feats_fc6,'fc6','fc6');
% before rotation: mean was 57
featureExtractorShallow = DeepFeatureExtractor(conf,true,16,'/home/amirro/storage/matconv_data/imagenet-vgg-s');
s40_feats_shallow =  featureExtractorShallow.extractFeaturesMulti(imgs,false,-90);

res_baseline_s40 = train_and_test_helper(s40_feats,s40_fra_classes(:),isTrain(:),[],train_params);
[ss,sm_baseline_s40] = summarizeResults(res_baseline_s40,s40_feats,train_params)

s40_feats_shallow_f = add_feature([],s40_feats_shallow,'fc6_s');

res_baseline_s40_shallow = train_and_test_helper(s40_feats_shallow_f,s40_fra_classes(:),isTrain(:),[],train_params);
[ss,sm_baseline_s40_shallow] = summarizeResults(res_baseline_s40_shallow,s40_feats_shallow_f,train_params)


% produceF a pretty plot.

aps = ss(2:end);

[r,ir] = sort(aps,'descend');
train_params.classNames'

train_params.classNames(class_types==1)
find(class_types==1)


mean(aps(class_types==UNDECIDED)) % 

mean(aps(class_types ~= NONTRANSITIVE))


%%
NONTRANSITIVE = 0;
SMALL = 1;
BIG = 2;
UNDECIDED = 3;
class_types = [0,...
    1,...
    1,...
    3,...
    0,...
    3,...
    3,...
    3,...
    1,...
    2,...
    3,...
    2,...
    2,...
    3,...
    2,...
    0,...
    3,...
    3,...
    2,...
    2,...
    1,...
    2,...
    1,...
    1,...
    2,...
    2,...
    2,...
    0,...
    3,...
    1,...
    1,...
    1,...
    2,...
    3,...
    2,...
    3,...
    2,...
    0,...
    2,...
    1];
%%

clf; h = figure(1); hold on;
% h1 = barh(r); ylim([0 41]);
% plot(aps,1:40,'go','LineWidth',3)
% ylim([0 41]);
% colors = distinguishable_colors(4);
set(gca,'YTickLabels',cellfun2(@(x)  strrep(x,'_',' '),train_params.classNames(ir)))
set(gca,'YTick',1:40)
    grid on;
colors = linspecer(4);
% colors = {'r','g','b','k'};
% colors = [1 0 0;0 1 0;0 0 1;0 0 0]
% for uuu = 0:3
%     barh(find(class_types(ir)==uuu),r(class_types(ir)==uuu),.8,'hist',colors{uuu+1});
%     dpc
% end
%%
for z = 1:length(ir)
    p =  barh(z,r(z));
    p.FaceColor =colors(1+class_types(ir(z)),:);    
end

% colors*255

% 
% for t = 0:3
%     pp = plot(.5,5,'r+');
%     pp.Color = colors(t+1,:);
% %     p.FaceColor = colors(t+1,:);
% end


% hhh = legend({'nontransitive','small','big','undecided'});
% 
% %dummies...

%%
xlabel('average precision');
% ylabel('class');
%%
saveTightFigure(gcf,'figures/s40.pdf')
%%
    

    
% for iClass = 1:length(classes)    
%     plot(perf_no_global(iClass).info.ap,classes(iClass),'bo','LineWidth',3);    
%     plot(perf_and_crop_and_global(iClass).info.ap,classes(iClass),'ro','LineWidth',3);
%     plot([0 perf_and_crop_and_global(iClass).info.ap], [classes(iClass) classes(iClass)],'k-');
% end

xlabel('Avg. precision');




%%
img = imread('VOC2012/JPEGImages/2011_006671.jpg');
imshow(img)


featureExtractor_cpu = DeepFeatureExtractor(conf,false,33);
featureExtractor = DeepFeatureExtractor(conf,true,33);

f_cpu = featureExtractor_cpu.extractFeatures(img);
f_gpu = featureExtractor.extractFeatures(img);

netPath = '/home/amirro/storage/matconv_data/imagenet-vgg-verydeep-16.mat';
net = load(netPath) ;
net.layers = net.layers(1:32);
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.normalization = net.meta.normalization;
imo = prepareForDNN({img},featureExtractor.net,false);
net.move('gpu');
net.eval({'input', gpuArray(imo)});
net.vars(end).precious=1;



scores = squeeze(gather(net.vars(net.getVarIndex('x32')).value));
scores = gather(net.vars(net.getVarIndex(predVar)).value);
f_gpu-f_cpu


