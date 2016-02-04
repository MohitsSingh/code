%S = all_results(1).surfaces;
% S = ff_mine;
for t= 1:length(S)
    % t = 5;  % INSERT TIME STEP
    % S = sufraces_in_timespace; % can change this to surfaces_in_time
    %= sufraces_in_timespace; % can change this to surfaces_in_time
    %     S = all_results(1).surfaces;
    imageDir  = 'Images_divided_by_maxValue';
    V = S(t).xyz;
    plot3(V(:,1),V(:,2),V(:,3),'.')
    [X,Y,Z] = world_to_surface(S(t).xyz);
    subplot(1,2,1)
    meshz(Z);
    imagesc(Z); colorbar
    xlim([110 310]);
    ylim([110 310]);
    set(gca, 'dataaspectratio', [1 1 1]);axis equal
    title(sprintf('%s%d', 'Calculated heights ', t))
    %
    % load('D:\Danny_Work\People\Graduate_Students\Itai\DataCu\LWC_1540.mat');    % t=2
    time = 1500 + 20*t;
    %addpath('D:\Danny_Work\People\Graduate_Students\Itai\DataCu')
    %     load (sprintf('%s%d', 'LWC_', time))
    %     load (sprintf('%s%d', 'U_', time))
    %     load (sprintf('%s%d', 'V_', time))
    %     load (sprintf('%s%d', 'W_', time))
    
    topo = zeros(480,480);
    topo_u = zeros(480,480);
    topo_v = zeros(480,480);
    topo_w = zeros(480,480);
    ZZ = LWC>.0001;
    for k=50:185
        topo(ZZ(:,:,k)) = k;
    end
    topo = topo*30;
    topo(topo<1860)=nan;
    %     subplot(1,2,2), imagesc(topo)
    
    %     set(gca, 'dataaspectratio', [1 1 1]); axis equal
    %     colorbar
    % saveas(fig1,'topo.png','png')
    %     title(sprintf('%s%d', 'True heights ', t))
    %     dpc;continue
    
    only_topo = topo>0;
%     Z(~only_topo) = nan;
    topo(~only_topo) = nan;
    subplot(1,2,1); imagesc(Z);
    axis equal; colorbar
    xlim([110 310]); ylim([110 310]);
    subplot(1,2,2); imagesc(topo);
    axis equal;colorbar
    xlim([110 310]); ylim([110 310]);
    dpc;
    %%
    %     dpc;
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
%
%
