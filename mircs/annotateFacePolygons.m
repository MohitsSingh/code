% load fra_db;
alternativeFacesDir = '/home/amirro/storage/data/Stanford40/annotations/faces_alt';
conf.get_full_image = true;
close all;
ids = {fra_db.imageID};
[ids,ir] = sort(ids);
fra_db = fra_db(ir);
for t = 1:length(fra_db)
    t    
    imgData = fra_db(t);
    if imgData.isTrain,continue,end
%     if imgData.classID==1,continue,end
    altPath = j2m(alternativeFacesDir,imgData);
    found_anno = false;
    found_new_anno = false;
    if exist(altPath,'file')
%         continue
        found_new_anno = true;
        load(altPath);
    else
        [objectSamples] = getGroundTruth(conf,{imgData.imageID},true);
        objectNames = cellfun2(@(x) x.name,objectSamples);
%         disp(objectNames)
        u = cellfun(@any,strfind(objectNames,'face'));
        if any(u)
            u = find(u,1,'first');
%             continue
            xy = [objectSamples{u}.polygon.x,objectSamples{u}.polygon.y];
            found_anno = true;
        end
    end                
    fprintf('new:%d\nold:%d\n',found_new_anno,found_anno);
    I = getImage(conf,imgData);
    clf; imagesc2(I);
    zoomToBox(inflatebbox(imgData.faceBox,3,'both',false));
    if found_anno
        plotPolygons(xy,'g-','Linewidth',2);
        title('found anno');
        drawnow
%         pause(.1)
    elseif found_new_anno
        plotPolygons(xy,'g-','Linewidth',2);
        title('found new anno');
        drawnow
%       
    else
        h = impoly();
        xy = getPosition(h);
        save(altPath,'xy');
    end        
%     dpc
end
