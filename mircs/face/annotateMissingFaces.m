% make sure face annotations exist for all image of interest.
% specific_classes_init;


% isValid = [newImageData.faceScore] >-.6;
isValid =true(size(newImageData));
trainOrTest = 'train';
groundtruth = consolidateGT(conf,'train');
groundtruth = [groundtruth,consolidateGT(conf,'test')];
groundtruth = groundtruth(strcmp('face',{groundtruth.name}));
imagesWithFace = ({groundtruth.sourceImage}); %assume one drinking face annotated per source image.
% although there may be a case or two this is not so, the first one
% annotated is probably the main face.

% A = cat(1,imagesWithFace{:});
% A = double(A); find(all(diff(A,1)==0,2))
% groundtruth(66)

% figure,imagesc2(getImage(conf,groundtruth(65).sourceImage));
% hold on; plotPolygons([groundtruth(66).polygon.x,groundtruth(66).polygon.y],'g--');

all_bbs = cell(size(newImageData));
annotationDir = '/home/amirro/storage/data/Stanford40/annotations/faces';
close all;
validIndices = 1:length(newImageData);
for ik = 1:length(validIndices)
    k = validIndices(ik);
%     k = 1
    
    % check if face annotation exists.
%     if (~class_labels(ik)),continue,end;
    
    
    
    currentID = newImageData(k).imageID;
    
    %if ~strcmp(currentID,'drinking_175.jpg'),continue,end
    [lia,locb]= ismember(currentID,imagesWithFace);
    newImageData(k).gt_face.poly = [];
    %     k
    %     clf;imagesc2(getImage(conf,newImageData(k)));hold on;plotBoxes(newImageData(k).gt_face.bbox,'g--');
    %     pause
    
    if (newImageData(k).faceScore > -.6)
        continue;
        
        %         bbb = newImageData(k).faceLandmarks.faceBox;
        %         bbb = bbb+I_rect([1 2 1 2]);
        %         clf;imagesc2(I);hold on;plotBoxes(bbb,'g--','LineWidth',3);
        
        %         drawnow;
        %disp('skipping...');continue;
    end
    %     newImageData(k).faceScore
    if (lia)
        locb = locb(1);
        newImageData(k).gt_face.poly = [groundtruth(locb).polygon.x,groundtruth(locb).polygon.y];
        newImageData(k).gt_face.bbox = pts2Box(newImageData(k).gt_face.poly);
        %         clf;imagesc2(getImage(conf,newImageData(k)));hold on;plotBoxes(newImageData(k).gt_face.bbox,'g--');
    else
        % check if bounding box exists in annotation file.
        fName = fullfile(annotationDir,[currentID '.txt']);
        needToAnnotate = true;
        if (exist(fName,'file'))
            [~,bb] = bbGt('bbLoad',fName);
            if (~isempty(bb))
                needToAnnotate = false;
            end
        end
        if (needToAnnotate)
            [I,I_rect] = getImage(conf,newImageData(k));
            clf; imagesc2(I);hold on;
            plotBoxes(I_rect,'m--','LineWidth',2);
            bbb = newImageData(k).faceLandmarks.faceBox;
            bbb = bbb+I_rect([1 2 1 2]);
            plotBoxes(bbb,'g--','LineWidth',3);            
            %             plotBoxes(newImageData(k).upperBodyDets(1,:),'r--','LineWidth',2);
            [~,api]=imRectRot('rotate',0);
            objs = bbGt( 'create', 1 );
            objs.lbl = 'face';
            bb = api.getPos();
            objs.bb = bb(1:4);
            bbGt( 'bbSave', objs, fName );
        end
        bb(3:4) = bb(3:4)+bb(1:2);
        
        newImageData(k).gt_face.bbox = bb;
        
        %error('continue implementing: add to newImageData');
    end
end
% end

% end

