--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Machine wrapping over the torchnet engine
--]]

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'

-- require 'utils/utils.lua'

paths.dofile('../util/Logger.lua')
local colors = paths.dofile('../util/ansicolors.lua')


local Machine = torch.class 'Machine'

--- Class that sets engine, criterion, model
-- @param opt
function Machine:__init(opt)
   opt = opt or {}
    if opt.GPU ~= -1 then
      -- Convert model to CUDA
      cutorch.setDevice(opt.GPU)
      print(colors.red .. 'Using gpu device: '.. opt.GPU..'\n'..colors.reset)
      cudnn.fastest = true
      cudnn.benchmark = true
   end

   self.trainDataset = opt.trainDataset -- training dataset to be used
   self.valDataset = opt.valDataset -- validation dataset to be used
   self.trainIters = opt.trainIters or -1 -- size of training dataset to be used

   self.validIters = opt.validIters  or -1 -- size of validation dataset to be used

   self.trainBatchSize = opt.trainBatch or 32
   self.valBatchSize = opt.validBatch or 32

   self.pose_feat_model, self.feat_ind = self:LoadPoseModel(opt)

   self.model,self.modelName = self:LoadModel(opt) -- model to be used
   self.criterion = self:LoadCriterion(opt) -- criterion to be used
   self.engine = self:LoadEngine(opt) -- engine to be used

   self.savePath = opt.save -- path where models has to be saved
   self.maxepoch = opt.nEpochs -- maximum number of epochs for training
   self.dataset = opt.dataset -- name of the base file used for training
   self.learningalgo = opt.optimMethod -- name of the learning algorithm used

   self.meters = self:LoadMeters(opt) -- table of meters, key is the name of meter and value is the meter
   self:attachHooks(opt)
   self:setupEngine(opt)
   self.snapshot = opt.snapshot
   self.numStages = opt.numStages
   self.nThreads = opt.nThreads
   self.trainlogger = Logger(paths.concat(opt.save, 'train.logger'), opt.continue)
   self.validlogger = Logger(paths.concat(opt.save, 'valid.logger'), opt.continue)
end

function Machine:LoadPoseModel(opt)
   local pose_feat_model = torch.load(opt.pose_feat_model)

   if opt.GPU ~= -1 then 
      print('==> Converting pose_feat_model to CUDA')
      pose_feat_model:cuda()
      cudnn.convert(pose_feat_model, cudnn)
   end
   local feat_ind = opt.feat_ind
   return pose_feat_model, feat_ind
end
--- Loads the model
-- @return Model loaded in CUDA,Name of the model
function Machine:LoadModel(opt)
   -- Continuing an experiment where it left off
   local model, name = 'default'
   if opt.continue  then
       local prevModel = opt.load .. '/final_model.t7'
       print('==> Loading model from: ' .. prevModel)
       model = torch.load(prevModel)

   -- Or a path to previously trained model is provided
   elseif opt.loadModel ~= 'none' then
       assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
       print('==> Loading model from: ' .. opt.loadModel)
       model = torch.load(opt.loadModel)
   -- Or we're starting fresh
   else
       print('==> Creating model from file: ' .. opt.netType .. '.lua')
       local netType = opt.netType .. '.lua'
       dofile(netType)
       model,name = createModel(opt)
             -- reset weights
      local method = 'xavier'
      model = dofile('../util/weight-init.lua')(model, method)
      print(colors.red..'Init model using '..method..colors.reset)
   end

   return model,name
end

--- Loads the criterion
-- @return Criterion loaded in CUDA
function Machine:LoadCriterion(opt)

   local criterion = nn.MSECriterion()

   if opt.GPU ~= -1 then
      -- Convert model to CUDA
      print('==> Converting model to CUDA')

      
      self.model:cuda()
      criterion:cuda()
      
      cudnn.convert(self.model, cudnn)
   end
   

   return criterion
end

--- Loads the engine
-- @return Optim Engine Instance
function Machine:LoadEngine(opt)
   local engine = tnt.OptimEngine()
   return engine
end

--- Loads all the meters
-- @return Table of meters such that, key is a string denoting meter name and value is the meter
-- Keys - Training Loss, Training Dice Score, Validation, Validation Dice Score, Param Norm, GradParam Norm, Norm Ratio, Time
function Machine:LoadMeters(opt)
   local meters = {}
   meters['Training Loss'] = tnt.AverageValueMeter()
   -- meters['Training Dice Score'] = tnt.AverageValueMeter()
   meters['Validation Loss'] = tnt.AverageValueMeter()
   -- meters['Validation Dice Score'] = tnt.AverageValueMeter()
   meters['Param Norm'] = tnt.AverageValueMeter()
   meters['GradParam Norm'] = tnt.AverageValueMeter()
   meters['Norm Ratio'] = tnt.AverageValueMeter()
   meters['Train MPJPE'] = tnt.AverageValueMeter()
   meters['Valid MPJPE'] = tnt.AverageValueMeter()
   meters['recons_err'] = tnt.AverageValueMeter()
   
   meters['Time'] = tnt.TimeMeter()
   return meters
end

--- Resets all the meters
function Machine:ResetMeters()
   for i,v in pairs(self.meters) do
      v:reset()
   end
end

--- Prints the values in all the meters
function Machine:PrintMeters()
   for i,v in pairs(self.meters) do
      io.write(('%s : %.10f | '):format(i,v:value()))
   end
end

--- Trains the model
function Machine:train()
   self.engine:train{
      network   = self.model,
      iterator  = getIterator('train',self.trainDataset,self.trainBatchSize, self.nThreads, self.trainIters),
      criterion = self.criterion,
      optimMethod = self.optimMethod,
      config = self.optimConfig,
      maxepoch = self.maxepoch
   }
end

--- Test the model against validation data
function Machine:test()
   self.engine:test{
      network   = self.model,
      iterator  = getIterator('test',self.valDataset,self.valBatchSize, self.nThreads, self.validIters),
      criterion = self.criterion,
   }
end

--- Given the state, it will save the model as ModelName_DatasetName_LearningAlgorithm_epoch_torchnet_EpochNum.t7
function Machine:saveModels(state)
   local savePath = paths.concat(self.savePath,('%s_%s_%s_epoch_torchnet_%d.t7'):format(self.modelName,self.dataset,self.learningalgo,state.epoch))
   torch.save(savePath,state.network:clearState())
   print(colors.blue..'\nSaved model to '..savePath..'\n'..colors.reset)
end

--- Adds hooks to the engine
-- state is a table of network, criterion, iterator, maxEpoch, optimMethod, sample (table of input and target),
-- config, optim, epoch (number of epochs done so far), t (number of samples seen so far), training (boolean denoting engine is in training or not)
-- https://github.com/torchnet/torchnet/blob/master/engine/optimengine.lua for position of hooks as to when they are called
function Machine:attachHooks(opt)

   --- Gets the size of the dataset or number of iterations
   local onStartHook = function(state)
      state.numbatches = state.iterator:exec('size')  -- for ParallelDatasetIterator
      -- if state.training and self.trainIters >= 1 then
      --    self.numbatches = math.min(state.numbatches,self.trainIters)
      -- elseif state.training ~= true and self.validIters >= 1 then 
      --    self.numbatches = math.min(state.numbatches, self.validIters)
      -- end
      -- print(state.numbatches)

   end

   --- Resets all the meters
   local onStartEpochHook = function(state)

      if self.learningalgo == 'sgd'  then
         state.optim.learningRate = self:LearningRateScheduler(state,state.epoch+1)
      end
      print(colors.red..("Epoch : %d, Learning Rate : %.10f "):format(state.epoch+1,state.optim.learningRate or state.config.learningRate)..colors.reset)
      self:ResetMeters()
      
      state.iterator:exec('resample')
   end

   --- Transfers input and target to cuda
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   local onSampleHook = function(state)
      -- print(state.sample.input:size())
      --- Transfers input and target to cuda
      igpu:resize(state.sample.input:size()):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      
      self.pose_feat_model:forward(igpu)
      igpu = self.pose_feat_model.modules[self.feat_ind].output
      --print(igpu:size())
      state.sample.input  = igpu
      state.sample.target  = tgpu
      -- state.network:zeroGradParameters()
      if state.network.forget then 
         state.network:forget()
      end
   end  -- alternatively, this logic can be implemented via a TransformDataset

   local onForwardHook = function(state)
      
   end

   --- Updates losses and dice score
   local onForwardCriterionHook = function(state)

      paths.dofile('../util/cal_mpjpe.lua')
      self.max_min = self.max_min or read_max_min()
      local preds = state.network.output:float()
      local labels =  state.sample.target:float()
      local value = cal_mpjpe(preds, labels, self.max_min, true)
      
      if state.training then
         self.meters['Training Loss']:add(state.criterion.output/state.sample.input[1]:size(1))
         self.meters['Train MPJPE']:add(value)

         if self.trainlogger then
           self.trainlogger:add{
               ['Train loss      '] = string.format("%.10f" , state.criterion.output/state.sample.input[1]:size(1)),
               
               ['MPJPE           '] = string.format("%.10f"   , value),
               ['LR        '] = string.format("%f"   , state.config.learningRate)
           }
         end
         -- self.meters['Training Dice Score']:add(CalculateDiceScore(state.network.output,state.sample.target))
      else
         self.meters['Validation Loss']:add(state.criterion.output/state.sample.input[1]:size(1))
         self.meters['Valid MPJPE']:add(value)
         -- self.meters['Validation Dice Score']:add(CalculateDiceScore(state.network.output,state.sample.target))
         if self.validlogger then
           self.validlogger:add{
               ['Valid loss      '] = string.format("%.10f" , state.criterion.output/state.sample.input[1]:size(1)),
               ['MPJPE           '] = string.format("%.10f"   , value)
           }
         end
      end
   end

   local onBackwardCriterionHook = function(state)
   end

   local onBackwardHook = function(state)
   end

   --- Update the parameter norm, gradient parameter norm, norm ratio and update progress bar to denote number of batches done
   local onUpdateHook = function(state)
      self.meters['Param Norm']:add(state.params:norm())
      self.meters['GradParam Norm']:add(state.gradParams:norm())
      self.meters['Norm Ratio']:add((state.optim.learningRate or state.config.learningRate)*state.gradParams:norm()/state.params:norm())
      xlua.progress(state.t,state.numbatches)
   end

   --- Sets t to 0, does validation and prints results of the epoch
   local onEndEpochHook = function(state)
      state.t = 0
      self:test()
      self:PrintMeters()
      if state.training  and self.snapshot ~= 0 and state.epoch % self.snapshot == 0 then
         self:saveModels(state)
      end
      
   end

   local onEndHook = function(state)
   end

   --- Attaching all the hooks
   self.engine.hooks.onStart = onStartHook
   self.engine.hooks.onStartEpoch = onStartEpochHook
   self.engine.hooks.onSample = onSampleHook
   self.engine.hooks.onForward = onForwardHook
   self.engine.hooks.onForwardCriterion = onForwardCriterionHook
   self.engine.hooks.onBackwardCriterion = onBackwardCriterionHook
   self.engine.hooks.onBackward = onBackwardHook
   self.engine.hooks.onUpdate = onUpdateHook
   self.engine.hooks.onEndEpoch = onEndEpochHook
   self.engine.hooks.onEnd = onEndHook
end

--- Returns the learning for the epoch
-- @param state State of the training
-- @param epoch Current epoch number
-- @return Learning Rate
-- Training scheduler that reduces learning by factor of 5 rate after every 4 epochs
function Machine:LearningRateScheduler(state,epoch)
    local decay = 0
    local step = 1
    decay = math.ceil((epoch - 1) / 4)
    return math.pow(0.1, decay)
end

--- Sets up the optim engine based on parameter received
-- @param opt It must contain optimMethod
function Machine:setupEngine(opt)
   if opt.optimMethod=='sgd' then
      self.optimMethod = optim.sgd
      self.optimConfig = {
         learningRate = opt.LR,
         momentum = opt.momentum,
         nesterov = true,
         weightDecay = opt.weightDecay,
         dampening = 0.0,
      }
   elseif opt.optimMethod=='adam' then
      self.optimMethod = optim.adam
      self.optimConfig = {
         learningRate = opt.LR,
         weightDecay = opt.weightDecay

      }
   elseif opt.optimMethod == 'adadelta' then
      self.optimMethod = optim.adadelta
      self.optimConfig = {
         learningRate = opt.LR,
         weightDecay = opt.weightDecay,
      }
   end
end

--- Iterator for moving over data
-- @param mode Either 'train' or 'valid', defines whether iterator for training or testing
-- @param ds Dataset for the iterator
-- @param size Size of data to be used
-- @param batchSize Batch Size to be used
-- @return parallel dataset iterator
function getIterator(mode,ds,batchSize, nThreads, iterSize)
   
   if iterSize == -1 then
      iterSize = ds:size()
   end
   return tnt.DatasetIterator{
      dataset = tnt.BatchDataset{
               batchsize = batchSize,
               dataset = tnt.ShuffleDataset{
                  dataset = ds,
                  size = iterSize
               }
            },
      transform =  ds.transforms

   }
end

