--[[
This code is part of Ultrasound-Nerve-Segmentation Program

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Main file
--]]

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'rnn'
tnt = require 'torchnet'
require 'opts'

torch.setnumthreads(1) -- speed up
torch.setdefaulttensortype('torch.FloatTensor')

--- Main execution script
function main(opt)
   
   -- loads the data loader
   require 'TempoH5Dataset'
   
   local trainDataset = tnt.TempoH5Dataset('train',opt)
   local valDataset = tnt.TempoH5Dataset('valid',opt)
   opt.trainDataset = trainDataset
   opt.valDataset = valDataset
   dofile(opt.machineType .. '.lua')
   local  m = TempoMachine(opt)
   
   m:train()
end

-- if not opt then
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

main(opt)
