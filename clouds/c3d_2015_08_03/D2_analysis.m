S = all_results(1).surfaces;
for t= 1:length(S)
    % t = 5;  % INSERT TIME STEP    
    % S = sufraces_in_timespace; % can change this to surfaces_in_time
     %= sufraces_in_timespace; % can change this to surfaces_in_time
%     S = all_results(1).surfaces;
    imageDir  ='Images_divided_by_maxValue';
    xyz = S(t).xyz;
    
    xyz(:,2:3) = xyz(:,2:3);
    x = xyz(:,1); y = xyz(:,2); z =xyz(:,3); % world coordinates
    dmap = xyz/30;
%     dmap(:,2:3) = dmap(:,2:3)+130;
    z_ = 1;
    rangeY=floor(min(dmap(:,2))):z_:ceil(max(dmap(:,2)));
    rangeX=floor(min(dmap(:,1))):z_:ceil(max(dmap(:,1)));
        
    % allow only x,y whose distance to originals is small enough
    
%     my_topo=zeros(480,480);
%     for k=50:185
%         topo(ZZ(:,:,k)) = k;
%     end
%     topo = topo*30;
%     
%     subplot(1,2,2), imagesc(topo)
%     xlim([110 310])
%     ylim([110 310])
    
    xyz_orig = xyz(:,1:2)/30;    
    [Y,X]=meshgrid(rangeX,rangeY);
    %Z = griddata(dmap(:,1),dmap(:,2),dmap(:,3),X,Y);
% %     offset = 130;
%     dmap(:,1:2) = dmap(:,1:2)+offset;
%     X = X+offset;
%     Y = Y+offset;
%     
    Z = griddata(dmap(:,1),dmap(:,2),dmap(:,3),X,Y);
    
    
    
% %     dists = l2([X(:) Y(:)],xyz_orig(:,1:2));
% %     dists = min(dists,[],2);
% %     dist_T = 100;
% %     Z(dists>dist_T) = min(Z(dists<dist_T));

    
%     Z_ = zeros(size(Y));
%     Z_(130:130+size(Z,1)-1,130:130+size(Z,2)-1) = Z;
%     X = X+130;
%     Y = Y+130;
    subplot(1,2,1)
%     surf(480-X,480-Y,Z*30,'EdgeColor','None');colorbar
    Z = Z*30;
    Z(Z<1860)=0;
    imagesc(Z); colorbar
    
%     view(0,90);
%     xlim([110 310]);
%     ylim([110 310]);
    set(gca, 'dataaspectratio', [1 1 1])
     title(sprintf('%s%d', 'Calculated heights ', t))
    
    %%
    % load('D:\Danny_Work\People\Graduate_Students\Itai\DataCu\LWC_1540.mat');    % t=2
    time = 1500 + 20*t;
    %addpath('D:\Danny_Work\People\Graduate_Students\Itai\DataCu')
%     load (sprintf('%s%d', 'LWC_', time))
%     load (sprintf('%s%d', 'U_', time))
%     load (sprintf('%s%d', 'V_', time))
%     load (sprintf('%s%d', 'W_', time))
    
    topo=zeros(480,480);
    topo_u=zeros(480,480);
    topo_v=zeros(480,480);
    topo_w=zeros(480,480);
    ZZ = LWC>.0001;
    for k=50:185
        topo(ZZ(:,:,k)) = k;
    end    
    topo = topo*30;
    
    subplot(1,2,2), imagesc(topo)
    xlim([110 310])
    ylim([110 310])
    set(gca, 'dataaspectratio', [1 1 1])
    colorbar
    % saveas(fig1,'topo.png','png')
    title(sprintf('%s%d', 'True heights ', t))
    %%
    dpc;
%     saveas(gca,sprintf('%s%d', 'surfaces_in_timespace_', t),'fig')
%     saveas(gca,sprintf('%s%d', 'surfaces_in_timespace_', t),'png')

    % fig2=figure;
    % imagesc(topo_u);
    % colorbar
    % title('topo u');
    %
    % fig3=figure;
    % imagesc(topo_v);
    % colorbar
    % title('topo v');
    %
    % fig4=figure;
    % imagesc(topo_w);
    % colorbar
    % title('topo w');
    
end