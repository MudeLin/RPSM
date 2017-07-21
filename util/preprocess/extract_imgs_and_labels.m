function obj = extract_imgs_and_labels(phase)


% Setup
addpaths;
close all;

if nargin < 1
    phase = 'val';
end
% Features{1} = H36MPose3DAnglesFeature();
Features{1} = H36MPose3DPositionsFeature('Monocular',true);
Features{2} = H36MPose2DPositionsFeature();
% Features{4} = H36MPose3DPositionsFeature('Monocular',true);
% Features{5} = H36MPose3DPositionsFeature('Symmetric',true);
% Features{6} = H36MPose2DPositionsFeature('Symmetric',true);
% Features{7} = H36MPose3DPositionsFeature('Monocular',true,'Symmetric',true);
% Features{8} = H36MPose3DAnglesFeature('Monocular',true);

% select the data
db = H36MDataBase.instance();
vidfeat = H36MRGBVideoFeature();


% train data
if strcmp(phase,'train')
    subjects = db.train_subjects;
elseif strcmp(phase,'test')
    subjects = db.test_subjects;
elseif strcmp(phase,'val')
    subjects = db.val_subjects;
else
    disp('Error, Unknow phase');
    disp(phase);
    return
end

img_folder = strcat('/data/human3.6m/linux_imgs/',phase,'/');
cropped_img_folder = strcat('/data/human3.6m/linux_cropped_imgs/',phase, '/');

if ~exist(img_folder, 'dir')
    mkdir(img_folder);
end

if ~exist(cropped_img_folder, 'dir')
    mkdir(cropped_img_folder);
end

img_list_folder = '/data/human3.6m/img_list'
if ~exist(img_list_folder, 'dir')
    mkdir(img_list_folder);
end


step = 5;

for camera_ind = [3,2,1,4]
    d2_outfile = fopen( strcat(img_list_folder,filesep,'linux_accv_',phase,'_camera_',num2str(camera_ind),'_label_2d.txt'),'w');
    d3_outfile = fopen( strcat(img_list_folder,filesep,'linux_accv_',phase,'_camera_',num2str(camera_ind),'_label_3d.txt'),'w');
    crop_d2_outfile = fopen(strcat(img_list_folder,filesep,'linux_accv_',phase,'_camera_',num2str(camera_ind),'_label_cropped_2d.txt'),'w');
    crop_d3_outfile = fopen( strcat(img_list_folder,filesep,'linux_accv_',phase,'_camera_',num2str(camera_ind),'_label_cropped_3d.txt'),'w');
    bbox_outfile = fopen( strcat(img_list_folder,filesep,'linux_accv_',phase,'_camera_',num2str(camera_ind),'bbox.txt'),'w');
    for subject = subjects
        
        %         if camera_ind == 1 && subject ~= 11
        %            continue
        %         end
        
        for action = db.actions
            if action == 1
                % pass Miscellaneous
                continue
            end
            if camera_ind == 1 && subject == 11 && action == 2
                continue
            end
            
            disp('processing subject,action');
            disp(strcat(num2str(subject),',',num2str(action)))
            
            for subaction = db.subactions
                newSubFolder = strcat(img_folder,'camera_',num2str(camera_ind),filesep,num2str(subject),filesep,num2str(action),'_',num2str(subaction));
                new_cropped_img_folder = strcat(cropped_img_folder,'camera_',num2str(camera_ind),filesep,num2str(subject),filesep,num2str(action),'_',num2str(subaction));
                
                if ~exist(new_cropped_img_folder, 'dir')
                    mkdir(new_cropped_img_folder);
                end
                if ~exist(newSubFolder, 'dir')
                    mkdir(newSubFolder);
                end
                Sequence = H36MSequence(subject,action,subaction,camera_ind,1:step:db.getNumFrames(subject, action, subaction));
                Subject = Sequence.getSubject();
                pos2dSkel = Subject.get2DPosSkel();
                posSkel   = Subject.getPosSkel();
                
                F = H36MComputeFeatures(Sequence, Features);
                
                [F{1},d3_body_skel] = Features{1}.select(F{1},posSkel,'body');
                [F{2},d2_body_skel] = Features{2}.select(F{2},pos2dSkel,'body');
                
                da = vidfeat.serializer(Sequence);
                FeatureName = 'ground_truth_bs';
                mask_feature_path = ['MySegmentsMat' filesep FeatureName filesep];
                load([Sequence.getPath() filesep mask_feature_path Sequence.getName '.mat'],'Masks');
                
                for frame = 1:step:db.getNumFrames(subject, action, subaction)
                    im = da.getFrame(frame);
                    img_fn = strcat(newSubFolder,filesep,num2str(frame),'.jpg');
                    img_fn = regexprep(img_fn, '\\','\\\\');
                    imwrite(im,img_fn);
                    crop_img_fn = strcat(new_cropped_img_folder,filesep,num2str(frame),'.jpg');
                    crop_img_fn = regexprep(crop_img_fn, '\\','\\\\');
                    d2_features = F{2}(int32(frame/step)+1,:);
                    
                    M = Masks{frame};
                    M = logical(M(:,:,1));
                    bbox = getBoundingBox(M);
                    
                    
                    crop_img = imcrop(im,bbox);
                    imwrite(crop_img,crop_img_fn);
                    
                    new_d2_feature = d2_features;
                    new_d2_feature(1:2:length(d2_features)) = d2_features(1:2:length(d2_features)) - bbox(1);
                    new_d2_feature(2:2:length(d2_features)) = d2_features(2:2:length(d2_features)) - bbox(2);
                    
                    bbox_label = regexprep(num2str(bbox),' +',',');
                    fprintf(bbox_outfile,strcat(img_fn,32,bbox_label,' \n'));
                    
                    d2_label = regexprep(num2str(F{2}(int32(frame/step) + 1, :)),' +',',');
                    fprintf(d2_outfile,strcat(img_fn,32,d2_label, ' \n') );
                    
                    d3_label = regexprep(num2str(F{1}(int32(frame/step) + 1,:)),' +',',');
                    fprintf(d3_outfile,strcat(img_fn,32,d3_label, ' \n') );
                    
                    new_d2_label = regexprep(num2str(new_d2_feature),' +',',');
                    fprintf(crop_d2_outfile,strcat(crop_img_fn,32,new_d2_label,' \n'));
                    
                    fprintf(crop_d3_outfile,strcat(crop_img_fn,32,d3_label, ' \n') );
                    
                    
                    
                    
                end
                clear('Masks','var');
            end
            
        end
    end
    fclose(d2_outfile);
    fclose(d3_outfile);
    
end
    function boundingbox = getBoundingBox(A)
        % arg: A, a logical matrix
        % return: a bounding box of the nonzeros in the matrix
        % [x_min y_min width height]
        % NOTE: this bounding box is different from image x,y coords
        [y,x] = ind2sub(size(A), find(A)); % matlab raw-base
        coord = [x, y];
        boundingbox = [min(coord) max(coord) - min(coord)];
        
    end

end



