

addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
addpath(genpath('~/code/utils'));
rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');
D = dir('~/TimeEvolution/*.png');

Is = {};

% calculate the flows
flows = struct('Vx',{},'Vy',{},'reliability',{},'DC',{});
for t = 1:length(D)
    Is{t} = imread(fullfile('~/TimeEvolution',D(t).name));
    if t > 1
        [Vx,Vy,reliab]=opticalFlow( Is{t-1}, Is{t},'resample',1,'type','LK');
        flows(t-1).Vx = Vx;
        flows(t-1).Vy = Vy;
        flows(t-1).reliability = Vx;
        flows(t-1).DC = [mean(Vx(:)) mean(Vy(:))];        
    end
end

%%
% create some figures
for t = 1:5:length(D)-1
    curFlow = flows(t);    
    I = imread(fullfile('~/TimeEvolution',D(t).name));    
    h = clf; subplot(1,2,1);imagesc2(I);
    d = 10;
    range_ = 1:d:400;
    [X,Y] = meshgrid(range_,range_);
   
    Vx = flows(t).Vx;
    Vy = flows(t).Vy;
    quiver(X,Y,Vx(range_,range_),Vy(range_,range_),0,'LineWidth',1);
    
    title('decimated')
    
    subplot(1,2,2);imagesc2(I);    
    Vx = flows(t).Vx;
    Vy = flows(t).Vy;
    quiver(Vx,Vy,'LineWidth',1);        
    title(['full. DC = ' num2str(flows(t).DC)]);    
%     dpc;
    
    drawnow;
    savefig([D(t).name(1:end-4) '_' D(t+1).name(1:end-4) '.fig']);
    %dpc
end


%%
%save flows.mat flows


%%

load flows.mat
dataDir = '~/TimeEvolution';
D = dir(fullfile(dataDir,'*.png'));
t = 1; % time to view
I = imread(fullfile(dataDir,D(t).name));

clf; % clear current figure

% get calculated u,v components of flows
Vx = flows(t).Vx;
Vy = flows(t).Vy;
subplot(1,2,1); 
imagesc(I); axis equal;hold on
quiver(Vx,Vy,'LineWidth',1); 
title('flow');
div = divergence(Vx,Vy);
subplot(1,2,2);
imagesc(div); title('divergence');axis equal




