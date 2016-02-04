function annotateMissingFaces2_helper(conf,fra_db,resDir,pTogether,forceAnnotate)
if (nargin < 4)
    pTogether = 3;
end
if (nargin < 5)
    forceAnnotate = false;
end
nTogether = pTogether^2;
% alreadyAnnotated = false(size(fra_db));
alreadyFound = false(size(fra_db));
fra_db = fra_db(randperm(length(fra_db)));
curFileInd = 1;
while(true && any(~alreadyFound))
    % select random subset
    % show all of them in subplot
    % choose a wrong one, if any
    nFilesFound = 0;
    filesFound = {}
    %
       
    while(nFilesFound < nTogether && curFileInd <= length(fra_db))
        curFileInd
        if (~alreadyFound(curFileInd))
            imgData = fra_db(curFileInd);
            
            fileExists =exist(j2m(resDir,imgData),'file');
            
            if (~fileExists)
                nFilesFound = nFilesFound+1;
                filesFound{end+1} = curFileInd;
            else                
                m =load(j2m(resDir,imgData));
                if (isempty(m.curImgData.faceBox))
                    nFilesFound = nFilesFound+1;
                    filesFound{end+1} = curFileInd;
                else
                    
                    %                 A = load(j2m(resDir,imgData));
                    %                 if (isempty(A.curImgData.faceBox))
                    %                     delete(j2m(resDir,imgData));
                    %                     nFilesFound = nFilesFound+1;
                    %                     filesFound{end+1} = curFileInd;
                    %                 else
                    alreadyFound(curFileInd) = true;
                    %                 end
                end
            end
        end
        curFileInd = curFileInd + 1;
    end
    filesFound = [filesFound{:}];
        
        if (nFilesFound==0)
            disp('all files annotated!');
            break;
        end
        
        m = ceil(sqrt(length(filesFound)));
        clf;
        imgDatas = fra_db(filesFound);
        all_good = true;
        clf;
        ha = tight_subplot(m,m,.1);
        Is = {};
        Rects = {};
        for ii = 1:length(filesFound)
            [Is{ii},Rects{ii}] = getImage(conf,imgDatas(ii));
        end
                                
        if (pTogether > 1)
            stillChoosing = 1;
            currently_selected = false(size(filesFound));
            while(stillChoosing)
                for ii = 1:length(filesFound)
                    axes(ha(ii));
                    I_rect = Rects{ii};
                    imagesc2(Is{ii}); plotBoxes(I_rect);title([num2str(ii),', ',...
                        strrep(imgDatas(ii).imageID,'_',' ')]);
                    xlim(I_rect([1 3]))
                    %     ylim([I_rect(2) mean(I_rect([2 4]))]);
                    ylim(I_rect([2 4]));
                    if (currently_selected(ii))
                        plotBoxes(imgDatas(ii).faceBox,'r','LineWidth',2);
                    else
                        plotBoxes(imgDatas(ii).faceBox,'m','LineWidth',2);
                    end
                end
                title('enter a number to specify an image to correct or SPACE to continue');
                [x,y,b] = ginput(1);
                sel_num = b-49+1;
                if (sel_num >=1 && sel_num <= length(filesFound))
                    currently_selected(sel_num) = true;
                else
                    stillChoosing = false;
                end
            end
            
        else
            currently_selected = 1;
        end
        for ii = 1:length(currently_selected)
            curImgData = imgDatas(ii);
            if (~currently_selected(ii))
                curPath = j2m(resDir,curImgData);
                
                save(curPath,'curImgData');
            else
                
                % end
                
                % for t = 1:length(fra_db)
                %     curImgData = fra_db(t);
                %     curImgData.imgIndex
                curPath = j2m(resDir,curImgData);
                if (exist(curPath,'file')),continue,end
                % %     if (exist(curPath,'file')),delete(curPath),end
                % %     continue
                [I,I_rect] = getImage(conf,curImgData);
                clf; imagesc2(I); plotBoxes(I_rect);
                I_rect = inflatebbox(I_rect,1.2,'both',false);
                xlim(I_rect([1 3]))
                %     ylim([I_rect(2) mean(I_rect([2 4]))]);
                ylim(I_rect([2 4]));
                
                
                % check if there is a 4x face detection...
                x4resPath = j2m('~/storage/s40_faces_baw_4x',curImgData);
                L = load(x4resPath);
                curImgData.faceBox = L.detections.boxes(1,:);
                %             if (curImgData.faceBox(end) < 1),continue,end
                plotBoxes(curImgData.faceBox,'m','LineWidth',2);
                
                if (~forceAnnotate)
                    
                    disp(strrep(curImgData.imageID,'_',' '));
                    disp('press SPACE to confirm, left mouse to correct the annotation,n to specify no visible faces');
                    [x,y,b] = ginput(1);
                    %     T = waitforbuttonpress;
                    curImgData.correct = false;
                    if (b==32) % accept face
                        curImgData.correct = true;
                        save(curPath,'curImgData');
                        continue;
                    end
                    if (b==110) % no face, record this
                        curImgData.faceBox = [];
                        save(curPath,'curImgData');
                        continue;
                    end
                    clc;
                    rects = {};
                    while(true)
                        disp('annotate correct faces');
                        rects{end+1}= getSingleRect();
                        disp('need more faces? SPACE to continue to next image, click mouse for more faces');
                        [x,y,k] = ginput(1);
                        if (k==32) %ESC
                            break;
                        end
                    end
                else
                    rects = {};
                    rects{end+1}= getSingleRect();
                end
                
                curImgData.faceBox = rects;
                save(j2m(resDir,curImgData),'curImgData');
            end
        end
    end