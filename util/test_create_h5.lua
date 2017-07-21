require 'hdf5'
require 'image'


function test_created_h5(gt_path, h5_path, skip)
	local myf = hdf5.open(h5_path, 'r')
	-- Open a file for read an test that it worked
	local fh,err = io.open(gt_path,'r')
	if err then print("OOps"); return; end
	lines = {}
	s_count = 1
	while true do
        line = fh:read()
        if line == nil then break end
        if s_count % 1 == 0 then 
	        table.insert(lines,line)
	    end
	    s_count = s_count + 1
    end
    print(#lines)
    local seq_start = 1
    for i = 1, s_count / skip do
    	local indice = myf:read('/indice/'..i):all()[1]
    	local imgid = myf:read('/image/'.. seq_start + i - 1):all()[1]
    	print('index at ' .. i)
        print(indice)
        print(lines[imgid])

    end
end


gt_path = '/data/human3.6m/std_img_list_3d/train_valid_data/valid_normalized_head_1.5w.txt'
output_h5_path = '../data/h3m/tempo_gan/valid_step_5.h5'
test_created_h5(gt_path,output_h5_path,5)

gt_path = '/data/human3.6m/std_img_list_3d/train_valid_data/train_normalized.txt'
output_h5_path = '../data/h3m/tempo_gan/train_all_step_5.h5'
test_created_h5(gt_path,output_h5_path,5)

