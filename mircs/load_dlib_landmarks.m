function fra_db = load_dlib_landmarks(conf,fra_db,inPath,outPath,debugging)

% if exist('dlib_lm.mat','file')
%     load dlib_lm.mat
%     dlib_landmarks.image_ids = image_ids;
%     dlib_landmarks.all_lm = all_lm;
% else
if nargin < 5
    debugging = false;
end
%fid = fopen('~/code/mircs/out2.txt');
%fid2= fopen('dlib_face_landmarks.txt');
fid = fopen(inPath);
fid2= fopen(outPath);
% file names...
s = fgetl(fid);
filePaths = {}
image_ids = {};
all_lm = {};
while s~=-1
    %         if (t >=1000),break,end
    filePaths{end+1} = s;
    f = fgetl(fid);
    nFaces = str2num(f);
    curLM = {};
    for u = 1:length(nFaces) % skip face lines
        f = fgetl(fid);
        LM = fgetl(fid2);
        LM = strrep(LM,'(','');
        LM = strrep(LM,')','');
        LM = strrep(LM,',','');
        LM = str2num(LM);
        LM = reshape(LM(1:end-2),2,[])';
        curLM{end+1} = LM;
    end
    
    curPath = filePaths{end};
    [~,name,ext] = fileparts(curPath);
    all_lm{end+1} = LM;
    image_ids{end+1} = [name ext];
    % %     curPath = fullfile(conf.imgDir,[name ext]);
    % %     I = imread(curPath);
    % %     clf; imagesc2(I);
    % %     plotPolygons(curLM,'g.');
    % %     dpc;
    s = fgetl(fid);
end
fclose(fid)
fclose(fid2)

%     save dlib_lm.mat image_ids all_lm

% end
landmark_ids = image_ids;
fra_ids = cellfun2(@lower,{fra_db.imageID});
[ints,lia,lib] = intersect(fra_ids,landmark_ids);

for t = 1:length(fra_db)
    fra_db_ind = lia(t);
    dlib_ind = lib(t);
    curLandmarks = all_lm{dlib_ind};
    fra_db(fra_db_ind).Landmarks_dlib = curLandmarks;
        
    if debugging
        clf; imagesc2(getImage(conf,fra_db(fra_db_ind)));
        bb = round(inflatebbox(pts2Box(curLandmarks),1.5,'both',false));
        %%plot_dlib_landmarks(curLandmarks);
        
        plotPolygons(curLandmarks,'g.');
        
        xlim(bb([1 3]));
        ylim(bb([2 4]));
        dpc
    end
    
    % % %     if isempty(strfind(lower(fra_db(fra_db_ind).imageID),'brush'))
    % % %         continue
    % % %     end
    % % %     clf; imagesc2(getImage(conf,fra_db(fra_db_ind)));
    % % % % %
    % % % % %
    % % %     bb = round(inflatebbox(pts2Box(curLandmarks),1.5,'both',false));
    % % % %     myMarkerSpec = {'g-','LineWidth',1};
    % % %     plot_dlib_landmarks(curLandmarks);
    % % %
    % % %     xlim(bb([1 3]));
    % % %     ylim(bb([2 4]));
    % % %     dpc
end