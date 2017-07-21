require 'paths'
require 'hdf5'
require 'xlua'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
root_folder = '/data/human3.6m/Release-v1.1/H36MDemo/'

resize_height = 368
resize_width = 368
label_width = 51
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


function createTempoImageIdH5(gt_path, h5_path, skip)
	label_width = 17 * 3 
	skip = skip or 1
	local myf = hdf5.open(h5_path, 'w')
	-- Open a file for read an test that it worked
	local fh,err = io.open(gt_path,'r')
	if err then print("OOps"); return; end
	lines = {}
	s_count = 1
	while true do
        line = fh:read()
        if line == nil then break end
        if s_count % skip == 0 then 
	        table.insert(lines,line)
	    end
	    s_count = s_count + 1
    end
    local last_seq = 'none'
	for i = 1, #lines do
		xlua.progress(i, #lines)
		local line = lines[i]
		local imgTensor = torch.FloatTensor(3,resize_height,resize_width)
		local labelTensor = torch.FloatTensor(label_width)
		elems = split(line,' ')
		fn = paths.concat(root_folder,elems[1])
		fn_elems = fn.split(fn,'/')
		local cur_seq = fn_elems[6] .. fn_elems[7]
		local indice = torch.FloatTensor(1)
		indice[1] = 1
		if cur_seq ~= last_seq then 
			last_seq = cur_seq
			indice[1] = 0
		end
		
		labels = split(elems[2],',')
		for l = 1, #labels do
			labelTensor[l] = tonumber(labels[l])
		end
		-- local img = image.load(fn)
		-- img = image.scale(img,resize_width,resize_height)
		local img_id = torch.IntTensor(1)
		img_id[1] = i * skip
		myf:write('/image/'..i, img_id)  -- record line id
		myf:write('/label/'..i, labelTensor)
		myf:write('/indice/'..i, indice)
		
		-- break
    end

	myf:close()
end



step = 5

gt_path = '/data/human3.6m/train_valid_data/train.txt'
output_h5_path = '../data/h3m/tempo/train_step_5.h5'
createTempoImageIdH5(gt_path,output_h5_path,step)

gt_path = '/data/human3.6m/train_valid_data/valid.txt'
output_h5_path = '../data/h3m/tempo/valid_step_5.h5'
createTempoImageIdH5(gt_path,output_h5_path,step)


phase = 'test'
for i = 1, 15 do
	gt_path = '/data/human3.6m/norm_img_list/'..phase..'_' .. tostring(i) .. '/filename_gt.txt'
	output_h5_path = '../data/h3m/tempo/'..phase..'_' .. tostring(i) .. '_step_' ..tostring(step).. '.h5'
	createTempoImageIdH5(gt_path,output_h5_path, step)
	print(output_h5_path..' done')
end

