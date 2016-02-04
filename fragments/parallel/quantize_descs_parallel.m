
addpath('~/code/SGE/');

defaultOpts;
tt_0_1x2_1000;

test_images = textread(sprintf(globalOpts.VOCopts.imgsetpath,globalOpts.VOCopts.testset),'%s');
train_images = textread(sprintf(globalOpts.VOCopts.imgsetpath,globalOpts.VOCopts.trainset),'%s');

all_images = [train_images;test_images];

code = ['cd /home/amirro/code/fragments; defaultOpts;tt_0_1x2_1000;init;'...
    'quantize_descs(globalOpts,img,vocab,kdtree);'];

to_calc = false(size(all_images));
cls = 1;

for k = 1:length(all_images)
    %     k
    fPath = getQuantFile(globalOpts,all_images{k});    
    if (~exist(fPath,'file'))
        to_calc(k) = true;
    end
end

% to_calc = (to_calc & [t1==1;t2==1]);

all_images = all_images(to_calc);

test_ = 0;

images = cell(0);

L = 100;
c = mod(1:length(all_images),L)+1;
for k = 1:L
    images{k} = all_images(c==k);
    %{images_(c==k)};
end

if (test_)
    images = images(1);
end

run_parallel(code, 'img',images,'-cluster', 'mcluster01');
% images = (length(train_images)+1):length(all_images);

% run_parallel(code, 'img',mat2cell(images,1,ones(1,length(images))),'-cluster', 'mcluster01');