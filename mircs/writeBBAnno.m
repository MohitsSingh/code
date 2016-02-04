function writeBBAnno(name,bb,baseDir)
n = size(bb,1);
objs = bbGt( 'create', n );
bbPath = j2m(baseDir,name,'.jpg.txt');
if (n > 0)
    % bb(:,3:4) = bb(:,3:4)-bb(:,1:2);
    bb = bb(:,1:4);
    
    for t = 1:n
        objs(t).lbl = 'hand';
        objs(t).bb = bb(t,:);
    end
end
bbGt( 'bbSave', objs, bbPath);