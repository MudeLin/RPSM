require 'torch'
require 'string'


local csv2tensor = require 'csv2tensor'

torch.setdefaulttensortype('torch.FloatTensor')

function range(from, to, step)
  step = step or 1
  return function(_, lastvalue)
    local nextvalue = lastvalue + step
    if step > 0 and nextvalue <= to or step < 0 and nextvalue >= to or
       step == 0
    then
      return nextvalue
    end
  end, nil, from - step
end

function read_max_min()
	local max_min_path = '../data/h3m/train_point_max_min.csv'
	local one_to_end = {} 
	for i in range(1,51,1) do
		one_to_end[i] = string.format("%d", i)
	end
	local max_min = csv2tensor.load(max_min_path,{include=one_to_end})
	-- print(max_min)
	return max_min
end

function un_max_min(value, max_min)
	assert(value:size(2) == max_min:size(2))
	for i = 1, value:size(1) do
		value[i] = torch.add(torch.cmul(value[i] , torch.csub(max_min[1] , max_min[2])) , max_min[2])
		-- print(value)
	end
	return value
end

function  cal_mpjpe(preds, labels, max_min, un_labels)
	local max_min = max_min or read_max_min() 
	local t_preds = preds:clone()
	local t_labels = labels:clone()
	local t_preds = un_max_min(t_preds, max_min):view(t_preds:size(1), 17, 3)
	un_labels = un_labels or false
	if un_labels then 
		t_labels = un_max_min(t_labels, max_min):view(t_labels:size(1), 17, 3)
	end
	local err = torch.csub(t_preds, t_labels)
	local joint_err = torch.squeeze(torch.mean(torch.norm(err,2,3), 1))
	-- print(joint_err)
	local ave_err = torch.mean(joint_err)
	-- print(ave_err)

	return ave_err, t_preds:view(t_preds:size(1),17 * 3)
end

function save_csv(path, data)
	local out = assert(io.open(path, "w"), 'can not open file : ' .. path)
	splitter = ","
	for i=1,data:size(1) do
		out:write(i .. ',')
	    for j=1,data:size(2) do
	        out:write(data[i][j])
	        if j == data:size(2) then
	            out:write("\n")
	        else
	            out:write(splitter)
	        end
	    end
	end
	out:close()
end
