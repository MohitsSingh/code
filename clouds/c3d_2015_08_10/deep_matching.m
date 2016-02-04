addpath('~/code/3rdparty/deepmatching_1.0.2_c++/');
baseImagesPath = '~/code/clouds/';
%%
addpath('/home/amirro/code/3rdparty/EpicFlow_v1.00/');
addpath('~/code/3rdparty/flow-code-matlab/');
% cmd = sprintf('./epicflow-static %s %s',p1,p2)
addpath('/home/amirro/code/3rdparty/SED/');
load('modelFinal.mat');

%%
path_to_deep_matching = '~/code/3rdparty/deepmatching_1.0.2_c++';
options = '';
path_to_epic_flow = '/home/amirro/code/3rdparty/EpicFlow_v1.00';
M = 'matched_pairs';
fileNames = dir(fullfile(M,'*.png'));
flows = struct('img1Name',{},'flow',{});

% for each file, look for a file with the same name but with t+1
N = 0;

fileNames = dir(fullfile(M,'T_*_01_05.png'));

for t = 1:length(fileNames)
    t
    im1Name = fullfile(M,fileNames(t).name);
    [pathstr,name1,ext] = fileparts(im1Name);
    p = strread(name1,'%s','delimiter','_');
    im2Name = fullfile(M,sprintf('%s_%s_%s_%s%s',p{1},p{2},p{4},p{3},ext))
    %u = strfind(name1,'_');
    %ind = str2num(name1(u(1)+1:u(2)-1));
    %if ~ismember(ind,[1 5 9 13]),
    %    continue
    %end
    %name2 = [name1(1:u),sprintf('%02.0f_%02.0f',ind+4,ind),];
    %im2Name = fullfile(M,[name2 ext]);
    if ~exist(im2Name,'file')
        continue
    end
    %im2Name = fullfile(M,fileNames(t+1).name);
    I1 = imread(im1Name);
    I2 = imread(im2Name);
    I = I1;
    if size(I,3)==1, I = cat(3,I,I,I); end;
    edgesFile = [fileNames(t).name '.edges'];
    if ~exist(edgesFile,'file')
        edges = edgesDetect(I, model); 
        fid=fopen(edgesFile,'wb'); 
        fwrite(fid,transpose(edges),'single'); 
        fclose(fid);
    end
    matchFile = [fileNames(t).name 'match.txt'];
%     if ~exist(matchFile,'file')
        deepMatchingCmd = sprintf('%s/deepmatching-static %s %s -png_settings -improved_settings -downscale 0 -out  %s',...
            path_to_deep_matching,im1Name,im2Name,matchFile);
        [status,result] = system(deepMatchingCmd);
%     end
    flow_out_file = [fileNames(t).name '.flo'];
%     if ~exist(flow_out_file,'file')
        epicflowCmd = sprintf('~/code/3rdparty/EpicFlow_v1.00/epicflow-static %s %s %s %s %s %s -a .5 -d 1',...
        im1Name,im2Name,'edges.bin',matchFile,flow_out_file,options);    
        [status,result] = system(epicflowCmd);
%     end
    N = N+1;
    flows(N).img1Name = im1Name;
    flows(N).img2Name = im2Name;
    if ~exist(flow_out_file,'file')
        continue
    end
    flows(N).flow = readFlowFile(flow_out_file);
end 

for t = 1:length(flows)
    flow = flows(t).flow;
    if ~isempty(flow)
        u = squeeze(flow(:,:,1));
        v = squeeze(flow(:,:,2));
        save([flows(t).img1Name '.mat'],'u','v');
    end
end    