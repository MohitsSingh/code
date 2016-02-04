% convert data base format:
function [NG,PS,PSTR] = convert_data_base_format_new(mirc_indx,method_indx,RES_ELEMENTS,MAX_WANTED_NEGATIVE)

    init_mircs;
    
    mirc=mircs(mirc_indx);
    
    NG = cell(length(method_indx),1);
    PS = cell(length(method_indx),1);
    BASE_INFO = '/home/liav/info/mircs/';
    BASE_RESULT = '/home/liav/data/mircs/';
%     BASE_RESULT = '/home/amir/ann_star_exp';
    for mi=1:length(method_indx)
        pair=method_indx{mi};

        methodi=pair(1);
        method=params.METHODS{methodi};
        exp_indx=pair(2);

        experiment=mirc.experiments(exp_indx);

        class_info_dir=sprintf('%s/%s',BASE_INFO,mirc.name);
        pos_fname=fullfile(class_info_dir,'fnames_.mat');
        load(pos_fname,'gt_train_idx','gt_test_idx');

        if isempty(RES_ELEMENTS)
            mirc_fname=fullfile(class_info_dir,fullfile('mirc',sprintf('%s.png',mirc.name)));
            RES_ELEMENTS=size(imread(mirc_fname));

        end
        params.RES_ELEMENTS=RES_ELEMENTS;

        base_out_dir=sprintf('%s/%s/',BASE_RESULT,mirc.name);
        base_out_dir=fullfile(base_out_dir,'images');

        full_method=method;
        if ~isempty(experiment.desc)
            full_method=[full_method '_' experiment.desc];
        end
        out_dir=fullfile(base_out_dir,full_method);

        %% positive
        dest_out=fullfile(out_dir,['C' num2str(3) num2str(3)]);
        pos_out_dir=fullfile(dest_out,'thresh_ov');
        load(fullfile(pos_out_dir,'data'),'data');
        pos_thresh=zeros(length(data),1);
        for j=1:length(data)
            pos_thresh(j)=data{j}.top_scores;
        end
        %positive_scores=pos_thresh(gt_test_idx)';
        positive_scores=pos_thresh';


        %% negative
        NC_DATA_DIR=fullfile(BASE_INFO,'data/pascal_NC');
        fname_name=fullfile(NC_DATA_DIR,sprintf('fnames_%s.mat',experiment.neg_class));
        load(fname_name);
        neg_names=fnames.test;
		
 		for j=1:length(neg_names)
 			neg_names{j}=strrep(neg_names{j},'/home/liav',BASE_HOME);
 		end
		
        neg_out_dir=fullfile(out_dir,'NC3');

        load(fullfile(neg_out_dir,'data'),'data');

        all_scores=[];
        all_names=[];
        all_frames=[];
        all_filesi=[];
        p=1;
        for filesi=1:length(data)
            if ~isempty(data{filesi})
                all_scores=[all_scores data{filesi}.top_scores];
                all_frames=[all_frames data{filesi}.top_frames];
                for j=1:length(data{filesi}.top_scores)
                    all_names{p}=neg_names{filesi};
                    all_filesi(p)=filesi;
                    p=p+1;
                end
            end
        end
        negative_scores=all_scores;


        %% get negative patches
        [sorted_score,si]=sort(negative_scores);

        rng=1:min(MAX_WANTED_NEGATIVE,length(negative_scores));
        neg_idx=si(rng);
        sorted_score=sorted_score(rng);
        negative_patches=get_patches(all_frames,[],neg_idx,all_names,params);

        NG{mi}={negative_patches,sorted_score};
        
        %% get positive patches
        gt_name=fullfile(class_info_dir,'gt.mat');
        load(gt_name);

        % get class test:
        PS{mi}={patches(gt_test_idx),positive_scores};
        % get class train:
        PSTR = patches(gt_train_idx);
        
    end

end
