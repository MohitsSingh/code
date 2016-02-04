function [pick] = ellipseNMS(ellipses_)
%ELLIPSENMS Summary of this function goes here
%   Detailed explanation goes here
    ellipseFeats = getEllipseFeatures(ellipses_);
   
    bboxes = cat(1,ellipseFeats.bbox);
    ovp =  boxesOverlap(bboxes);
    ovp = max(ovp,eye(size(ovp)));
    
    for k = 1:size(ovp,1)
        jj = find(ovp(k,:) > .8);        
        xy = ellipseFeats(k).xy;
        plot(xy(:,1),xy(:,2),'r'); 
        
                for q = 2:length(jj)
                    ee = ellipseFeats(jj(q));
                    xy1 = ee.xy;
                    d1 = l2(xy,xy1);
                    d1 = max(min(d1,[],2));
                    
                        
                    
%             q
%             clf; plot(xy(:,1),xy(:,2),'r'); 
%             hold on;
%             ee = ellipseFeats(jj(q));
%             plot(ee.xy(:,1),ee.xy(:,2),'g');
%             pause;
        end
        
%         hold on;
%         for q = 2:length(jj)
%             q
%             clf; plot(xy(:,1),xy(:,2),'r'); 
%             hold on;
%             ee = ellipseFeats(jj(q));
%             plot(ee.xy(:,1),ee.xy(:,2),'g');
%             pause;
%         end
    end
end

