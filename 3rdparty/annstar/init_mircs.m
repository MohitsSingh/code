BASE_HOME = '\\math03-lx\liav';%'~liav';
BASE_INFO = [BASE_HOME '/info/mircs'];
BASE_RESULT = [BASE_HOME '/data/mircs'];


NC_DATA_DIR=fullfile(BASE_INFO,'data/pascal_NC');
params.METHODS={'L2','ExemplarSVM','SVM','Felzenszwalb','L2NN','L2NN_trans','SVM_trans','none'};
COLORS={{'b','g','r','c','m','k'},{'','','y','k'}};

clear mirc mircs
mircs=struct('name',{},'type',{},'notes',{},'short_notes',{},'experiments',{});

mirc.name='Picture1';
mirc.type='Suit';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_person','imagenet_but_suit'},'desc',{'','ImageNet only'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture1_highvar';
mirc.type='Suit';
mirc.notes='Suit, high variability of TP';
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_person','imagenet_but_suit'},'desc',{'','ImageNet only'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture3';
mirc.type='Eye';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_bird_cat_cow_dog_horse_person_sheep','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture6';
mirc.type='Eye';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_bird_cat_cow_dog_horse_person_sheep','pascal+imagenet_but_eye'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture6_sub';
mirc.type='Eye';
mirc.notes='Eye sub-MIRC';
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_bird_cat_cow_dog_horse_person_sheep','pascal+imagenet_but_eye_submirc'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture6_sub_gray';
mirc.type='Eye';
mirc.notes='Eye train MIRC, test sub-MIRC';
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_bird_cat_cow_dog_horse_person_sheep','desc','','methods',3:4);
mircs(end+1)=mirc;

mirc.name='Picture7';
mirc.type='Eyeglasses';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_person','pascal+imagenet_but_eyeglasses'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture7_highvar';
mirc.type='Eyeglasses';
mirc.notes='Eyeglasses, high variability of TP';
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_person','pascal+imagenet_but_eyeglasses'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture9';
mirc.type='Bicycle';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_bicycle_old','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture9_highvar';
mirc.type='Bicycle';
mirc.notes='Bicycle, high variability of TP';
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_bicycle','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture10';
mirc.type='Fly';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture12';
mirc.type='Aeroplane';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_aeroplane','pascal+imagenet_but_aeroplane'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture12_sub';
mirc.type='Aeroplane';
mirc.notes=[mirc.type ' sub-MIRC'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_aeroplane','pascal+imagenet_but_aeroplane_submirc'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture12_sub_gray';
mirc.type='Aeroplane';
mirc.notes=[mirc.type ' train MIRC, test sub-MIRC'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_aeroplane','desc','','methods',3:4);
mircs(end+1)=mirc;

mirc.name='Picture15';
mirc.type='Ship';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_boat','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture16';
mirc.type='Horse';
mirc.notes=[mirc.type ' head'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_cow_horse','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture16_hires';
mirc.type='Horse';
mirc.notes=[mirc.type ' head high resolution'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_cow_horse','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture16_hires_ear';
mirc.type='Horse';
mirc.notes=[mirc.type ' ear high resolution'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_cow_horse','all_but_cow_horse'},'desc',{'','edges'},'methods',{3,3});
mircs(end+1)=mirc;

mirc.name='Picture21';
mirc.type='Eagle';
mirc.notes=mirc.type;
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_bird','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture22';
mirc.type='Horse';
mirc.notes=[mirc.type ' legs'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_cow_horse','all_but_cow_horse'},'desc',{'','5 submodels'},'methods',{1:length(params.METHODS),4});
mircs(end+1)=mirc;

mirc.name='Picture22_sub';
mirc.type='Horse';
mirc.notes=[mirc.type ' legs sub-MIRC'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_cow_horse','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture22_sub_gray';
mirc.type='Horse';
mirc.notes=[mirc.type ' legs train MIRC, test sub-MIRC'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_cow_horse','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

mirc.name='Picture23';
mirc.type='Horse';
mirc.notes=[mirc.type ' torso'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_cow_horse','pascal+imagenet_but_horse'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture23_sub';
mirc.type='Horse';
mirc.notes=[mirc.type ' torso sub-MIRC'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class',{'all_but_cow_horse','pascal+imagenet_but_horse_submirc'},'desc',{'','PASCAL+ImageNet'},'methods',{1:length(params.METHODS),3});
mircs(end+1)=mirc;

mirc.name='Picture23_sub_gray';
mirc.type='Horse';
mirc.notes=[mirc.type ' torse train MIRC, test sub-MIRC'];
mirc.short_notes=mirc.notes;
mirc.experiments=struct('neg_class','all_but_cow_horse','desc','','methods',1:length(params.METHODS));
mircs(end+1)=mirc;

%requires TTYPE=2; params.LOWRES_RAT=[1 1]; params.felz_non_flipped=false;
%and modification to output_full_images not to blur the image 
mirc.name='Picture25_gray';
mirc.type='Horse';
mirc.notes=[mirc.type ' Felzenszwalb trained PASCAL class, test horse head MIRC'];
mirc.short_notes=mirc.notes;
% all_but_cow_horse is not enough
mirc.experiments=struct('neg_class','all_but_bird_cat_cow_dog_horse_person_sheep','desc','','methods',4);
mircs(end+1)=mirc;

%same requirements as above
mirc.name='Picture25_house';
mirc.type='Horse';
mirc.notes=[mirc.type ' Felzenszwalb trained PASCAL class, test horse head MIRC placed nearby a house'];
mirc.short_notes=mirc.notes;
% all_but_cow_horse is not enough
mirc.experiments=struct('neg_class','all_but_bird_cat_cow_dog_horse_person_sheep','desc','','methods',4);
mircs(end+1)=mirc;

%same requirements as above
mirc.name='Picture25_half';
mirc.type='Horse';
mirc.notes=[mirc.type ' Felzenszwalb trained PASCAL class, test half a horse placed nearby a house'];
mirc.short_notes=mirc.notes;
% all_but_cow_horse is not enough
mirc.experiments=struct('neg_class','all_but_bird_cat_cow_dog_horse_person_sheep','desc','','methods',4);
mircs(end+1)=mirc;

%same requirements as above
mirc.name='Picture25_full';
mirc.type='Horse';
mirc.notes=[mirc.type ' Felzenszwalb trained PASCAL class, test full horse placed nearby a house.\nAP=82, while on full PASCAL test (much more positives and harder) reported AP=42\n'];
mirc.short_notes='Felzenszwalb trained PASCAL class, test full horse';
% all_but_cow_horse is not enough
mirc.experiments=struct('neg_class','all_but_bird_cat_cow_dog_horse_person_sheep','desc','','methods',4);
mircs(end+1)=mirc;