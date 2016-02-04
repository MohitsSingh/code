function polys = compressMasks(masks)
%
polys = {};
tic_id = ticStatus('compressing masks',.2,.1);

% profile off
for ii = 1:length(masks)
    M = masks{ii};
    %     B = fliplr(bwtraceboundary2(M));
    B = bwboundaries(M);
    if isempty(B)
        continue
    end
    B1 = fliplr(B{1});
    %         x2(poly2mask2(B,size2(M))-M)
    %         simplifyPolygon
    %     B1 = simplifyPolygon(B1,1);
    polys{ii} = B1;
    tocStatus(tic_id,ii/length(masks))
    continue;
    
    %continue
    clf; subplot(1,2,1);
    imagesc2(M);
    plotPolygons(B,'r-','LineWidth',3);
    size(B1,1)/size(B,1)
    plotPolygons(B1,'g-+');
    subplot(1,2,2); imagesc2(xor(poly2mask2(B,size2(M)),poly2mask2(B1,size2(M))));
    continue;
    dpc
    %         if ii > 5
    %             break
    %         end
end
% profile viewer
