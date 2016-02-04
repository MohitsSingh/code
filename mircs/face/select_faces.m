function select_faces(conf,fra_db,resDir,pTogether)
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
            
            fileExists = exist(j2m(resDir,imgData),'file');
            
            if (~fileExists)
                nFilesFound = nFilesFound+1;
                filesFound{end+1} = curFileInd;
            else
%                 m =load(j2m(resDir,imgData));
%                 if (isempty(m.curImgData.faceBox))
%                     nFilesFound = nFilesFound+1;
%                     filesFound{end+1} = curFileInd;
%                 else
                    
                    %                 A = load(j2m(resDir,imgData));
                    %                 if (isempty(A.curImgData.faceBox))
                    %                     delete(j2m(resDir,imgData));
                    %                     nFilesFound = nFilesFound+1;
                    %                     filesFound{end+1} = curFileInd;
                    %                 else
                    alreadyFound(curFileInd) = true;
                    %                 end
%                 end
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
            title('enter a number to specify a bad image, or SPACE to continue');
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
        validFace = false;
        curPath = j2m(resDir,curImgData);
        if (~currently_selected(ii))
            validFace = true;
            
        end
        save(curPath,'validFace');
    end
end