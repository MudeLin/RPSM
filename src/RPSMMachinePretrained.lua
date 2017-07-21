require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'math'
-- require 'utils/utils.lua'
require 'machine'

local colors = paths.dofile('../util/ansicolors.lua')


local TempoMachine = torch.class('TempoMachine','Machine')

--- Class that sets engine, criterion, model
-- @param opt
function TempoMachine:__init(opt)
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

   self.sharedModel = self:LoadSharedModel(opt)
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
   self.rho = opt.rho 
   self.np = opt.np
   self:AddMeters(opt)
   self.trainlogger = Logger(paths.concat(opt.save, 'train.logger'), opt.continue)
   self.validlogger = Logger(paths.concat(opt.save, 'valid.logger'), opt.continue)
end
function TempoMachine:AddMeters(opt)
   for k = 1, opt.rho do
      self.meters['Train MPJPE at stage ' .. k] = tnt.AverageValueMeter()
      self.meters['Valid MPJPE at stage ' .. k] = tnt.AverageValueMeter()
      
   end
end
--- Loads the criterion
-- @return Criterion loaded in CUDA
function TempoMachine:LoadCriterion(opt)
   require 'rnn'

   local criterion = nn.ParallelCriterion(true)
   for k = 1, opt.rho do
      local stage_criterion = nn.SequencerCriterion(nn.MSECriterion())
      criterion:add(stage_criterion, 1.0)
   end
   if opt.rho == 1 then 
      criterion = nn.SequencerCriterion(nn.MSECriterion())
   end
   
   if opt.GPU ~= -1 then
      -- Convert model to CUDA
      self.model:cuda()
      criterion:cuda()

   end

   return criterion
end
function TempoMachine:LoadSharedModel(opt)
   print('loading shared model '..opt.sharedModel)
   local sharedModel = torch.load(opt.sharedModel)
   if opt.GPU ~= -1 then 
      print('==> Converting shared model to CUDA')
      sharedModel:cuda()
      cudnn.convert(sharedModel,cudnn)
   end
   return sharedModel
end

--- Returns the learning for the epoch
-- @param state State of the training
-- @param epoch Current epoch number
-- @return Learning Rate
-- Training scheduler that reduces learning by factor of 10 rate after every 4 epochs
function TempoMachine:LearningRateScheduler(state,epoch)
    local decay = 0
    local step = 1
    if epoch % 4 == 0 then 
       decay = math.pow(0.1, math.ceil((epoch - 1) / 4))
       print(colors.red .. 'decrease leraing rate by '.. math.pow(0.1, decay).. colors.red)
       return state.config.learningRate * decay
    else 
       return state.config.learningRate
    end
end

--- Adds hooks to the engine
-- state is a table of network, criterion, iterator, maxEpoch, optimMethod, sample (table of input and target),
-- config, optim, epoch (number of epochs done so far), t (number of samples seen so far), training (boolean denoting engine is in training or not)
-- https://github.com/torchnet/torchnet/blob/master/engine/optimengine.lua for position of hooks as to when they are called
function TempoMachine:attachHooks(opt)

   --- Gets the size of the dataset or number of iterations
   local onStartHook = function(state)
      state.numbatches = state.iterator:exec('size')  -- for ParallelDatasetIterator
      -- if state.training and self.trainIters >= 1 then
      --    self.numbatches = math.min(state.numbatches,self.trainIters)
      -- elseif state.training ~= true and self.validIters >= 1 then 
      --    self.numbatches = math.min(state.numbatches, self.validIters)
      -- end
      -- print(state.numbatches)
      if not state.training then 
         print('\n validating \n')
         
      end
   end

   --- Resets all the meters
   local onStartEpochHook = function(state)
      if self.learningalgo == 'sgd'  then
         state.config.learningRate = self:LearningRateScheduler(state,state.epoch+1)
      end
      print(colors.red..("Epoch : %d, Learning Rate : %.10f "):format(state.epoch+1,state.config.learningRate or state.config.learningRate)..colors.reset)
      self:ResetMeters()
      
   end

   --- Transfers input and target to cuda
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   local onSampleHook = function(state)
      igpu:resize(state.sample.input:size()):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      self.sharedModel:forward(igpu)
      state.sample.input  = self.sharedModel.modules[opt.feat_ind].output 
      state.sample.target  = tgpu
      state.network:forget()
   end
   local onForwardHook = function(state)
      if not state.training then
         xlua.progress(state.t,state.numbatches)
      end
   end

   --- Updates losses and dice score
   local onForwardCriterionHook = function(state)

      paths.dofile('../util/cal_mpjpe.lua')
      self.max_min = self.max_min or read_max_min()

      local preds = nil
      if self.rho > 1 then 
         preds = state.network.output[self.rho]:float():narrow(2,1,self.np*3):contiguous()
      else
         preds = state.network.output:float():narrow(2,1,self.np*3):contiguous()
      end

      local labels = state.sample.target:float():narrow(2,1,self.np*3):contiguous()
      local value = cal_mpjpe(preds, labels, self.max_min, true)
      
      if state.training then
         self.meters['Train MPJPE']:add(value)
         self.meters['Training Loss']:add(state.criterion.output / state.sample.input[1]:size(1))
         if self.trainlogger then
           self.trainlogger:add{
               ['Train loss      '] = string.format("%.10f" , state.criterion.output/state.sample.input[1]:size(1)),
               
               ['Train MPJPE           '] = string.format("%.10f"   , value),
               ['LR        '] = string.format("%f"   , state.config.learningRate)
           }
         end
         
      else
         self.meters['Valid MPJPE']:add(value)
         self.meters['Validation Loss']:add(state.criterion.output/state.sample.input[1]:size(1))
         -- self.meters['Validation Dice Score']:add(CalculateDiceScore(state.network.output,state.sample.target))
         if self.validlogger then
           self.validlogger:add{
               ['Valid loss      '] = string.format("%.10f" , state.criterion.output/state.sample.input[1]:size(1)),
               ['Valid MPJPE           '] = string.format("%.10f"   , value)
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

function getIterator(mode,ds,batchSize, nThreads, iterSize)
   
   if iterSize == -1 then
      iterSize = ds:size()
   end
   return tnt.DatasetIterator{
               dataset = ds,
               transform =  ds.transforms

   }
end

