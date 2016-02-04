%load ~/code/mircs/s40_fra.mat;
load ~/storage/mircs_18_11_2014/s40_fra.mat
nImages = length(s40_fra);
top_face_scores = zeros(nImages,1);
for t = 1:nImages
    top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
end
min_face_score = 0;
img_sel_score = (top_face_scores > min_face_score);img_sel_score = img_sel_score(:);
sel_train =  [s40_fra.isTrain];
%fra_db_new = fra_db;
fra_db = s40_fra(img_sel_score(:) & sel_train(:));
goodFaces = false(size(fra_db));
newFaceDataDir = '/home/amirro/storage/data/Stanford40/annotations/faces_oct_31_14';
ensuredir(newFaceDataDir);
%%

% alreadyAnnotated = false(size(fra_db));
%
% % initialize notAnnotated
% for u = 1:length(alreadyAnnotated)
%     curPath = j2m(newFaceDataDir,fra_db(u));
%     if (exist(curPath,'file'))
%         alreadyAnnotated(u) = true;
%     end
% end
%%

% annotateMissingFaces2_helper(conf,fra_db,newFaceDataDir);
fra_db = s40_fra(~img_sel_score(:));% & ~sel_train(:));
nTogether = 1,forceAnnotate = false;
conf.get_full_image = 1;
annotateMissingFaces2_helper(conf,s40_fra,newFaceDataDir,nTogether,forceAnnotate);

% make another pass, making sure that everything is in place....

load ~/code/mircs/s40_fra_faces_d.mat

viewingResDir = '/home/amirro/storage/data/Stanford40/annotations/faces_viewed_nov_13_11';
ensuredir(viewingResDir);
test_set = ~[s40_fra_faces_d.isTrain];
nTogether = 3;
viewFaces(conf,s40_fra_faces_d(test_set),viewingResDir,nTogether);

f_test = find(test_set);
for u = 1:length(f_test)
    u
    k = f_test(u);
    L = load(j2m(viewingResDir,s40_fra_faces_d(k)));
    s40_fra_faces_d(k).faceBox = L.curImgData.faceBox;
end

for t = 1:length(s40_fra_faces_d)
    if (iscell(s40_fra_faces_d(t).faceBox))
        B = s40_fra_faces_d(t).faceBox{1};
        s40_fra_faces_d(t).faceBox = B(1,1:4);
    end
end

save ~/code/mircs/s40_fra_faces_d.mat s40_fra_faces_d
selectionDir = '/home/amirro/storage/data/Stanford40/annotations/valid_faces';
ensuredir(selectionDir)
select_faces(conf,s40_fra_faces_d(~test_set),selectionDir,3);

%
% annotateMissingFaces2_helper(conf,s40_fra_faces_d(test_set),viewingResDir,nTogether,false);
%


%% 
I = imread('~/storage/data/Stanford40/JPEGImages/reading_005.jpg');
x2(I);


% first, find out where is all the data saved.



% annotateDatabase



