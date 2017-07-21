--[[
	input to the coarse model is the img feat that extracted by 
		2d pose estimation model that will be of greatly unbiased to the train set and test set
	output of the model coarse predict 3d pose prediction
--]]


require 'nn'
require 'nngraph'
require 'rnn'
require 'cudnn'
require 'cutorch'

function feature_adaption_module(opt, k)
	-- # k^th stage
	local inp = nn.Identity()()
	local d3_feat = nn.Sequential()
	d3_feat:add(cudnn.SpatialConvolution(opt.num_ch, opt.num_ch, 5,5, 2,2, 2,2))
	d3_feat:add(cudnn.ReLU(true))
	d3_feat:add(cudnn.SpatialConvolution(opt.num_ch, 128, 3,3, 1,1, 1,1))
	d3_feat:add(cudnn.ReLU(true))
	d3_feat:add(cudnn.SpatialMaxPooling(2,2,2,2):ceil())
	d3_feat:add(nn.View(-1,128 * 12 * 12))  -- 7 * 7
	d3_feat:add(nn.Linear(128 * 12 * 12,opt.img_feat_dim))
	d3_feat:add(cudnn.ReLU(true))
	local img_feat = d3_feat(inp)
	return nn.gModule({inp}, {img_feat})
end

function d2_pose_module(opt, k)
	local inps = {}
	inps[1] = nn.Identity()()
	local d2_feat = nn.Sequential()
	if k > 1 then 
		inps[2] = nn.Identity()()
		d2_feat:add(nn.JoinTable(2))
		d2_feat:add(cudnn.SpatialConvolution(opt.num_ch + opt.num_ch, opt.num_ch, 3, 3, 1, 1, 1, 1, 1))
	else
		d2_feat:add(cudnn.SpatialConvolution(opt.num_ch, opt.num_ch, 3, 3, 1, 1, 1, 1, 1))
	end 
	d2_feat:add(cudnn.ReLU(true))
	
	d2_feat:add(cudnn.SpatialConvolution(opt.num_ch, opt.num_ch, 3, 3, 1, 1, 1, 1, 1))
	d2_feat:add(cudnn.ReLU(true))
	local pose_feat = d2_feat(inps)

	return nn.gModule(inps,{pose_feat})
end


function createModel(opt)
	local seq_inps = nn.Identity()() 
	-- input is feature maps from pretrained network
	local cnn_feat = seq_inps

	local rpsm = nn.Sequential()
	local outputs = {}
	local pose_aware_outputs = {}

	for k = 1, opt.rho do 
		local pose_aware_feat 
		if k > 1 then 
			pose_aware_feat = d2_pose_module(opt,k){cnn_feat, pose_aware_outputs[k - 1]}
		else
			pose_aware_feat = d2_pose_module(opt,k){cnn_feat}
		end
		local d3_pose_feat = feature_adaption_module(opt,k)(pose_aware_feat)
		-- joint input 
		local rnn
		if k > 1 then 
			local rnn_inputs = nn.JoinTable(2){d3_pose_feat, outputs[k - 1]}
		    rnn = nn.Sequencer(
				nn.Sequential()
				:add(nn.FastLSTM(opt.img_feat_dim + opt.np * 3, opt.hiddenSize))
				:add(nn.Linear(opt.hiddenSize, opt.np * 3))
			)(rnn_inputs)
			
		else
			local rnn_inputs = d3_pose_feat
			rnn = nn.Sequencer(
				nn.Sequential()
				:add(nn.FastLSTM(opt.img_feat_dim, opt.hiddenSize))
				:add(nn.Linear(opt.hiddenSize, opt.np * 3))
			)(rnn_inputs)
		end

		outputs[k] = rnn
		pose_aware_outputs[k] = pose_aware_feat
	end
	
	local model = nn.gModule({seq_inps}, outputs)

 	return model, 'rpsm_pertrained'
end


-- -- Test 
-- require 'cutorch'
-- require 'nn'
-- require 'cunn'
-- require 'cudnn'
-- torch.setdefaulttensortype('torch.FloatTensor')
-- opt = {}
-- opt.np = 17
-- opt.num_ch = 128
-- opt.inputRes = 368
-- opt.hiddenSize = 1024
-- opt.img_feat_dim = 1024
-- opt.rho = 3

-- local inp = torch.Tensor(10,3,opt.inputRes, opt.inputRes):cuda()

-- model = createModel(opt)
-- graph.dot(model.fg,'rmsp','./fg')

-- model:cuda()

-- local out = model:forward(inp)
-- print(out[1]:size())
-- freeMemory, totalMemory = cutorch.getMemoryUsage(1)
-- print(freeMemory)
-- print(totalMemory)

-- -- -- Test