function [res,best_ovp] = findCoveringSegment(regions,target,ucm)

[ovp,ints,uns] = regionsOverlap2(regions,target);

% out of the non-zero overlaps, align masks, but only if a sufficient
% overlap was found.

f_ovp = find(ovp > .2);
if (~any(f_ovp))
    res = [];best_ovp = max(ovp);
    return;
end

% regions_orig = regions(f_ovp);
regions = regions(f_ovp);
% M = regionAdjacencyGraph(regions);
% groups = enumerateGroups(M,2);
% [~,regions] = expandRegions(regions,2,groups,M);
% do an exhaustive search over some translations.
target_scale = 2*(nnz(target)/pi)^.5; % if it were a circle
max_T = target_scale/5;
% [dx,dy] = meshgrid(floor(-max_T/2):2:ceil(max_T/2));
[yy,xx] = find(target); bbox = pts2Box(xx,yy);
target_ = cropper(target,bbox);
% ovps = zeros(size(dx));
% bests = zeros(size(dx));

% turn everything into polygons
region_polys = cellfun2(@(x) fliplr(bwtraceboundary2(x)),regions);
% region_polys_reduced = {};
polys = struct('x',{},'y',{},'hole',{});
reduce_t = .1;
for k = 1:length(region_polys)
    xy = region_polys{k};
    x = xy(:,1); y = xy(:,2);
    [x1 y1] = reducem(x,y,reduce_t);
    polys(k).x = x1; polys(k).y = y1; polys(k).hole = 0;
    region_polys{k} = [x1 y1];  
end

target_poly = fliplr(bwtraceboundary2(target));
 x = target_poly(:,1); y = target_poly(:,2);
 [x1 y1] = reducem(x,y,reduce_t);
 target_poly_r = [x1 y1];

x0 = [0 0];
[x_out,fval,exitflag,output] = fminsearch(@(x) targetOverlap(polys,target_poly_r,x),x0);
% [x_out,fval,exitflag,output] = fminsearch(@(x) targetOverlap_ucm(ucm,target_poly_r,x),x0);
x_out = round(x_out);
[x,y] = transformPolygon(target_poly_r,x_out);
bboxShifted = bbox+x_out([1 2 1 2]);
bestTarget = shiftRegions(target_,bboxShifted,regions{1});
% bestRegion = regions{bests(ib)};
res = bestTarget;
best_ovp = -fval;

    function v = targetOverlap(polys,target_poly,x_t)
        
        debug_ = true;
        [x,y] = transformPolygon(target_poly,x_t);       
        ints = zeros(size(polys));uns = zeros(size(polys));
        curPoly.x= x;curPoly.y =y; curPoly.hole = 0;
        for u = 1:length(polys)
            P_int = PolygonClip(curPoly,polys(u),1);
            for t = 1:length(P_int)
                ints(u) = ints(u)+polyarea(P_int(t).x,P_int(t).y);
            end
            P_un = PolygonClip(curPoly,polys(u),3);
            for t = 1:length(P_un)
                uns(u) = uns(u) + polyarea(P_un(t).x,P_un(t).y);
            end
        end
        ovp_ = ints./uns;
        [v,iv] = max(ovp_);
        
        if (debug_)
            clf;
            x_t
            plotPolygons(target_poly,'r-');
            plotPolygons([x y],'g-');
            plotPolygons([polys(iv).x,polys(iv).y],'m--');
            axis image; drawnow
            pause(.01)
        end
     
        v = -v;
        %v = -v*exp(-norm(x_t)^2/100);
    end

    function [x,y] = transformPolygon(target_poly,x_t)
        x = target_poly(:,1)+x_t(1);y = target_poly(:,2)+x_t(2);
    end
end


