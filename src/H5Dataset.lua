--[[
This code is part of Ultrasound-Nerve-Segmentation Program

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Data Loader used to load nerve segmentation data
--]]

require 'torchnet'
require 'hdf5'
t = dofile ('../util/transforms.lua')
local tnt = require 'torchnet.env'

local H5Dataset,parent = torch.class('tnt.H5Dataset','tnt.Dataset', tnt)

function H5Dataset:__init(set, opt)
    parent.__init(self)
    self:setup(set, opt)
end

function H5Dataset:setup(set, opt)
    self.h5_path = opt.h5_path or opt[set..'H5Path']
    print('setting up H5Dataset '..self.h5_path)
    self.h5data = hdf5.open(self.h5_path,'r')

    if set == 'train' then
        self.nsample = opt.trainIters
        self.transforms = trainTransforms(opt)
    else 
        if opt.validIters == nil or opt.validIters == -1 then 
            local xf = self.h5data:all()
            self.nsample = 0
            for i,v in pairs(xf['image']) do
                self.nsample = self.nsample + 1
            end
        else
            self.nsample = opt.validIters
        end

        self.transforms = validTransforms(opt)
    end
    print('Total img number '..self.nsample)
    -- body

end


function H5Dataset:get(idx)
    local img = self.h5data:read('/image/'..idx):all()
    img:csub(0.5)
    -- img = img:mul(256):add(-128)
    local label = self.h5data:read('/label/'..idx):all()
    local visulize = false 
    if visulize == true then 
        require 'image'
        img_w = image.display{image=img, win = img_w}
        require 'sys'
        print(label)
        sys.sleep(1.0)
    end
    return {input = img, target = label}
    -- body
end

function H5Dataset:size()
    return self.nsample
end


--- Returns transform function used for training
function trainTransforms(opt)
    local f = function(sample)
        local images = sample.input
        local transforms = t.Compose{
            t.RandomScale(opt.minScale, opt.maxScale),
            t.RandomCrop(opt.inputRes)
        }
        local imagesTransformed = torch.Tensor(images:size(1),3,opt.inputRes,opt.inputRes)
        for i=1,images:size(1) do
            imagesTransformed[i] = transforms(images[i]:float())
            -- require 'image'
            -- img_w = image.display{image=imagesTransformed[i], win = img_w}
            -- sys.sleep(1.0)
        end
        sample['input'] = imagesTransformed
        return sample
    end
    return f
end

--- Returns validation function used for training
function validTransforms(opt)
    local f = function(sample)
        local images = sample.input
        local transforms = t.Compose{
            t.RandomCrop(opt.inputRes, opt.inputRes)
        }
        local imagesTransformed = torch.Tensor(images:size(1),3,opt.inputRes,opt.inputRes)
        for i=1,images:size(1) do
            imagesTransformed[i] = transforms(images[i]:float())
        end
        sample['input'] = imagesTransformed
        return sample
    end
    return f
end

-- Test
--  local set = 'train'
--  local opt = {}
--  opt.trainH5Path = '../data/h3m/train_3k.h5'

-- dt = tnt.H5Dataset(set, opt)
-- print(dt:get(1))

-- Test
