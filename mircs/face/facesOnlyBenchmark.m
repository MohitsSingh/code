% faces only dataset.

load s40_fra.mat
s40_fra_faces = s40_fra;
conf.get_full_image = true;


for t = 1:length(s40_fra_faces)
    t
    
    if (s40_fra_faces(t).isTrain && s40_fra_faces(t).indInFraDB~=-1)
        s40_fra_faces(t) = switchToGroundTruth(s40_fra_faces(t))
    else % use manually annotated face but remove landmarks
        s40_fra_faces(t).mouth = [];
        s40_fra_faces(t).raw_faceDetections = [];
        s40_fra_faces(t).faceBox_gt = [];
        s40_fra_faces(t).mouth_gt = [];
        L = load(j2m('/home/amirro/storage/data/Stanford40/annotations/faces_oct_31_14',s40_fra_faces(t)));
        s40_fra_faces(t).faceBox = L.curImgData.faceBox;
        %            [I_orig,I_rect] = getImage(conf,s40_fra_faces(t));
        %         clf;imagesc2(I_orig); plotBoxes(s40_fra_faces(t).faceBox);
        %         pause;continue
        I_box = s40_fra_faces(t).faceBox;
        if iscell(I_box)
            I_box = I_box{1};
        end
        I_box =I_box(1,1:4);
        [I_orig,I_rect] = getImage(conf,s40_fra_faces(t));
        orig_box = s40_fra(t).raw_faceDetections.boxes(1,1:4)+I_rect([1 2 1 2]);
        %     s40_fra(k).raw_faceDetections.boxes(1,1:4)-I_box
        if sum(abs(I_box-orig_box)) > 0 % fix the box
            I_box(3:4) = I_box(3:4)+I_box(1:2);
            s40_fra_faces(t).faceBox = I_box;
            %             clf;imagesc2(I_orig); plotBoxes(s40_fra_faces(t).faceBox);
            %         pause;
        end
        
        
        
    end
end
%%
save ~/code/mircs/s40_fra_faces.mat s40_fra_faces


%% fix the ucm results
d = dir('~/storage/faces_only_landmarks/*.mat');
figure,plot(sort([d.bytes]))   
examineImg(conf,'cleaning_the_floor_010.jpg',s40_fra_faces_d)
L = load('-mat','~/storage/faces_only_seg/cleaning_the_floor_010.mat.error');