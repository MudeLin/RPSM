--[[
This code is part of Ultrasound-Nerve-Segmentation Program

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Main file
--]]
require 'paths'
-- paths.dofile('setup.lua')     -- Parse command line input and do global variable initialization
require 'H5Dataset'
require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
tnt = require 'torchnet'
require 'sys'
torch.setnumthreads(1) -- speed up
torch.setdefaulttensortype('torch.FloatTensor')

-- command line instructions reading
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 context encoder training script')
cmd:text()
cmd:text('Options:')
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-dataset',        'mpii', 'Dataset choice: mpii | flic')
cmd:option('-dataDir',       '../data', 'Data directory')
cmd:option('-expDir',         '../exp',  'Experiments directory')
cmd:option('-expID',          'default', 'experiment id')
cmd:option('-manualSeed',         -1, 'Manually set RNG seed')
cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
cmd:option('-nThreads',            2, 'Number of data loading threads')
cmd:option('-lstmModel',          '/home/geek/data/model/trained_model/pose_lstm_mpii_adam_epoch_torchnet_8.t7','Path of the model to be used')
cmd:option('-h5_path',          '../data/h3m/test_1_step_12.h5', 'Name of validation file')
cmd:option('-saveCSVPath',          'none', 'path for csv save path')
cmd:option('-saveMPJPEPath',           'none',   'path for save mpjpe result')
cmd:option('-actionID'     ,         1,          'specified action id ')


cmd:option('-validIters',1000,'Size of the batch to be used for validation')
cmd:option('-savePath','../exp/mpii/result','Path to save models')
cmd:text(' ---------- Data options ---------------------------------------')
cmd:text()
cmd:option('-numStages',          1, 'number of cpm stages')
cmd:option('-np',                 17, 'number of joints')
cmd:option('-inputRes',          368, 'Input image resolution')
cmd:option('-outputRes',          46, 'Output heatmap resolution')
cmd:option('-sigma',              20, 'sigma for generate gaussian')
cmd:option('-visualize',          false, 'Whether visualize')
cmd:option('-no_substract_mean',      false, 'whether to substract 0.5 to [-.5, 0.5]')
cmd:option('-sharedModel',        '',        'pretrained shared model')
cmd:option('-refineModel',        'none', 'pose feature model')
cmd:option('-rho',                 3, 'rpsm stages')
cmd:option('-feat_ind',            33)
cmd:option('-maxFrames',         10,  'max frames to capture information')

cmd:option('-validImgList',         '','')
cmd:option('-rootImgFolder',        '', '')
-- cmd:option('-optimMethod','sgd','Algorithm to be used for learning - sgd | adam')
-- cmd:option('-maxepoch',250,'Epochs for training')
-- cmd:option('-cvParam',2,'Cross validation parameter used to segregate data based on patient number')

require 'rnn'
require 'nngraph'
local M = paths.dofile('../util/transforms.lua')
paths.dofile('../util/cal_mpjpe.lua')
require 'TempoH5Dataset'
--- Main execution script
function demo(opt)
    if opt.GPU ~= -1 then
      -- Convert model to CUDA
      cutorch.setDevice(opt.GPU)
      print( 'Using gpu device: '.. opt.GPU..'\n')

      cudnn.fastest = true
      cudnn.benchmark = true
   end
   
   print('Loading shared model '.. opt.sharedModel)
   local sharedModel = torch.load(opt.sharedModel)
   sharedModel:evaluate()
   
  print('load refine model ' .. opt.refineModel)
   local refineModel = torch.load(opt.refineModel)
   -- refineModel:remember('eval')
   refineModel:evaluate()

   print('Loading refineModel ' .. opt.refineModel)
   local valDataset = tnt.TempoH5Dataset('valid',opt)
   local dataset_size = valDataset:size(true)
   local preds = {} 
   for i = 1,opt.rho do
      preds[i] = torch.Tensor(dataset_size, opt.np*3)
   end
   local labels = torch.Tensor(dataset_size, opt.np*3)
   
   local i = 1
   while i < dataset_size do
      
      xlua.progress(i, dataset_size)

      local sample = valDataset:get(i, true)
      local input = sample.input
      local target = sample.target
      if i + input:size(1) > dataset_size then
         input = input:narrow(1,1, dataset_size - i)
         target = target:narrow(1,1, dataset_size - i)
      end

      local imagesTransformed = torch.Tensor(input:size(1),3,opt.inputRes,opt.inputRes)
      for k=1,input:size(1) do
         imagesTransformed[k] = M.CenterCrop(opt.inputRes)(input[k]:float())
      end
 	   sharedModel:forward(imagesTransformed:cuda())
      local out = sharedModel.modules[opt.feat_ind].output 
      local output = refineModel:forward(out)

      target = target:view(-1, opt.np *3)

      for j = 1, target:size(1) do 
         labels[i + j - 1] = target[j]
         if opt.rho > 1 then 
            for k = 1,opt.rho do
               preds[k][i + j - 1]  = output[k][j]:float()
            end
         else
            preds[1][i + j - 1]  = output[j]:float()
         end
         
      end
      
      i = i + target:size(1)
      collectgarbage()

      if opt.visualize then 
         input_w = image.display{image=imagesTransformed[1],win=input_w}
         sys.sleep(1.0)
      end
   end

   
   local value, pred_labels = cal_mpjpe(preds[opt.rho], labels)
   print('MPJPE is ' .. tostring(value))
   if (opt.saveCSVPath ~= 'none') then
      save_csv(opt.saveCSVPath  .. '.csv', pred_labels)
      print('Result saved to '.. opt.saveCSVPath  .. '.csv')
   end
   if (opt.saveMPJPEPath ~= 'none') then 
      local save_path = opt.saveMPJPEPath .. '.csv'
      local f=io.open( save_path  ,"a+")
      
      assert(f ~= nil, ' can not open file at ' .. save_path)
      if opt.actionID == 1 then 
         f:write('\n')
      end 
      if opt.actionID == 15  then -- 15 is max action 
         f:write(value..'\n')
      else
         f:write(value..',')
      end
      f:close()
      print('save mpjpe result to ' .. save_path)
   end

end

local opt = cmd:parse(arg or {}) -- Table containing all the above options
opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
demo(opt)
