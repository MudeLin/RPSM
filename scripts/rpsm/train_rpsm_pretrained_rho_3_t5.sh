
cd ../../src
th main_tempo_rpsm.lua -GPU 1 -netType ../models/rpsm/rpsm_pretrained \
-trainH5Path ../data/h3m/tempo/train_step_5.h5 -validH5Path ../data/h3m/tempo/valid_step_5.h5 \
-trainIters 52971 -validIters 3000 \
-maxFrames 5 \
-expID rpsm_1024_rho3_t5 \
-optimMethod adam \
-inputRes 368 \
-nEpochs 200 \
-LR 1e-4 \
-np 17 \
-feat_ind 33 \
-machineType RPSMMachinePretrained \
-sharedModel ../models/torch_model/caffe_d2_pose_module_shared.t7 \
-trainImgList /data/human3.6m/train_valid_data/train.txt \
-validImgList /data/human3.6m/train_valid_data/valid.txt \
-rootImgFolder /data/human3.6m/code-v2.1/Release-v1.1/H36MDemo \
-img_feat_dim 1024 \
-hiddenSize 1024 \
-weightDecay 1e-4 \
-rho 3 

