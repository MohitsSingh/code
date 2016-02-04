function [regions,region_sel] = getRegionSubset(conf,currentID,maxRegions,occludersOnly)
if (nargin < 4)
    occludersOnly = false;
end

if (maxRegions <= 1)
    T_ovp = maxRegions;
    maxRegions = inf;
else
    T_ovp = .5;
end

[regions,regionOvp,G] = getRegions(conf,currentID,false);

if (occludersOnly) 
    occludersPath = fullfile(conf.occludersDir,strrep(currentID,'.jpg','.mat'));
    L = load(occludersPath);
    region_sel = L.region_sel;
    if (isempty(region_sel))        
        region_sel = false(size(regions));
        return;
    else    
        occlusionPatterns = L.occlusionPatterns;
        t = find(region_sel);
        region_sel(t) = region_sel(t) & [occlusionPatterns.seg_in_face] >0 & ...
            [occlusionPatterns.face_in_seg] < .5 & [occlusionPatterns.seg_in_face] < 1;
        regionOvp = regionOvp(region_sel,region_sel);
        regions = regions(region_sel);
    end
end

region_sel = suppresRegions(regionOvp, T_ovp);
region_sel = region_sel{1};
% region_sel = col(find(region_sel));
region_sel = vl_colsubset(region_sel,maxRegions,'random');
regions = regions(region_sel);
end