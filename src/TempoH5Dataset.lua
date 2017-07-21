

require 'torchnet'
require 'hdf5'
-- t = dofile ('../util/transforms.lua')
local tnt = require 'torchnet.env'
require 'H5Dataset'
require 'paths'
local TempoH5Dataset,parent = torch.class('tnt.TempoH5Dataset','tnt.H5Dataset', tnt)



local function split(s, sep)
    local parts, off = {}, 1
    local first, last = string.find(s, sep, off, true)
    while first do
        table.insert(parts, string.sub(s, off, first - 1))
        off = last + 1
        first, last = string.find(s, sep, off, true)
    end
    table.insert(parts, string.sub(s, off))
    return parts
end



function TempoH5Dataset:readImgList(img_list_path, rootfolder)
    print('Reading image list : ' .. img_list_path)
    local namesFile = io.open(img_list_path)
    local idx = 1
    local totalFns = {}
    -- t = sys.clock()
    for line in namesFile:lines() do
        -- sys.tic()
        local fn = split(line, " ")[1]
        fn = paths.concat(rootfolder, fn)
        table.insert(totalFns, fn)
        -- t = sys.toc()
        -- print(t)
    end
    print('Total image list : ' .. #totalFns)
    return totalFns
end

function TempoH5Dataset:__init(set, opt)
    parent.__init(self,set, opt)
    self.maxFrames = opt.maxFrames
    self.np = opt.np 
    self.inputRes = opt.inputRes
    self.img_list = self:readImgList(opt[set .. 'ImgList'], opt.rootImgFolder)
    self.substract_mean = not opt.no_substract_mean
    self.gan_label_level = opt.gan_label_level -- 1 for image level , 2 for sequence level 
end

-- get max frames at idx location
function TempoH5Dataset:get(idx, not_random_index )
    idx = not_random_index  and idx or torch.random(1, self.nsample)
    -- print(idx)
    local seq_start = idx
    local seq_end = idx + self.maxFrames - 1
    if seq_end > self.nsample  then 
        seq_end = self.nsample 
    end
    for i = seq_start + 1, seq_end do
        local indice = self.h5data:read('/indice/'..i):all()[1]
        if indice < 0.5 then 
            seq_end = i - 1
            break
        end
    end

    local img_width = self.inputRes
    local seq_len = seq_end - seq_start + 1

    local imgs = torch.Tensor(seq_len, 3,img_width, img_width)
    local labels = torch.Tensor(seq_len, self.np*3)
    for i = 1, seq_len do 

        local imgid = self.h5data:read('/image/'..seq_start + i - 1):all()[1]
        
        local img = image.load(self.img_list[imgid])
        img = image.scale(img, img_width, img_width)
        imgs:narrow(1,i,1):copy(img)        
        labels:narrow(1,i,1):copy(self.h5data:read('/label/'..idx + i - 1):all())
    end
    if self.substract_mean then 
        imgs:csub(0.5) -- map to [-.5, .5]
    end
    return {input = imgs, target = labels}


    -- body
end

function TempoH5Dataset:size(not_random_index)
    return not_random_index and self.nsample or math.floor(self.nsample / self.maxFrames)
end


--- Returns transform function used for training
function trainTransforms(opt)
    local f = function(sample)
        local images = sample.input
        local transforms = t.Compose{
            t.RandomScale( opt.minScale, opt.maxScale),
            t.RandomCrop(opt.inputRes)
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

--- Returns validation function used for training
function validTransforms(opt)
    local f = function(sample)
        -- local images = sample.input
        -- local transforms = t.Compose{
        --     t.RandomCrop(opt.inputRes, opt.inputRes)
        -- }
        -- local imagesTransformed = torch.Tensor(images:size(1),3,opt.inputRes,opt.inputRes)
        -- for i=1,images:size(1) do
        --     imagesTransformed[i] = transforms(images[i]:float())
        -- end
        -- sample['input'] = imagesTransformed
        return sample
    end
    return f
end

--Test
--  local set = 'train'
--  local opt = {}
--  opt.trainH5Path = '../data/h3m/tempo_gan/valid_step_5.h5'
--  opt.trainIters = 3000
--  opt.maxFrames = 10
--  opt.inputRes = 368
--  opt.np  = 17
--  opt.trainImgList = '/data/human3.6m/std_img_list_3d/train_valid_data/valid_normalized_head_1.5w.txt'
--  opt.rootImgFolder = '/data/human3.6m/code-v1.1/Release-v1.1/H36MDemo/'
-- dt = tnt.TempoH5Dataset(set, opt)
-- print(dt:get(1))

-- Test