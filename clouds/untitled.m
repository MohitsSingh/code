 %%  
        
        for tt = -50:10:50
            tt
            ppp = Points;
            ppp(:,3) = ppp(:,3)+tt*30;
            p_test = cam.project(ppp(:,[1 2 3]));
            %         II = im2double(Img{17-t+1});
            
            II = im2double(Img{t});
            II = imrotate(II,270);
            clf; imagesc2(II.^.25);
            plotPolygons(p_test,'r.');
            dpc
        end