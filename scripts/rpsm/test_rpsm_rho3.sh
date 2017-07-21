
cd ../../src
num_action=15
phase='test'
root_folder='/data/human3.6m/norm_img_list'
step=5
for ((aid = 1; aid <= $num_action; ++aid))
do
	# echo ../data/h3m/$phase\_${aid}_step_12.h5
	th evaluate_tempo_rpsm.lua -GPU 1 -sharedModel ../models/torch_model/caffe_d2_pose_module_shared.t7 \
	-validIters -1  -h5_path ../data/h3m/tempo/$phase\_${aid}\_step\_${step}.h5 \
	-rootImgFolder /data/human3.6m/code-v2.1/Release-v1.1/H36MDemo \
	-validImgList $root_folder/${phase}\_${aid}/filename_gt.txt \
	-inputRes 368 \
	-np 17 \
	-feat_ind 33 \
	-refineModel ../exp/h3m/rpsm_1024_rho3_t5/rpsm_pertrained_h3m_adam_epoch_torchnet_50.t7 \
	-saveCSVPath $root_folder/${phase}\_${aid}/rho_3_t5_predict_rpsm\_${step} \
	-saveMPJPEPath '../mpjpe/h3m/rpsm_rho_3_t5_cont_pretrained_mpjpe' \
	-actionID $aid \
	-maxFrames 5 \
	-rho 3 \
	  2>&1 | tee ../log/$phase\_${aid}\_${step}.log
done

