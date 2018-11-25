require 'nn'
require 'sys'
require 'torch'
require 'os'

local Trainer = torch.class('Trainer')

-- Perform one epoch of training with predefined batches.
function Trainer:train(train_data, train_labels, model, criterion, optim_method, layers, state, params, grads)
  model:training()

  local num_batches = 0
  local train_size = 0

  for key, batch in pairs(train_data) do
    num_batches = num_batches + 1
    train_size = train_size + batch:size(1)
  end

  local timer = torch.Timer()
  local time = timer:time().real
  local total_err = 0

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end

  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local config -- for optim
  if opt.optim_method == 'adadelta' then
    config = { rho = 0.95, eps = 1e-6 } 
  elseif opt.optim_method == 'adam' then
    config = {}
  end

  -- shuffle batches
  local shuffle = torch.randperm(num_batches)
  for i = 1, shuffle:size(1) do
    local splits = 1
    local nSent = train_data[tostring(shuffle[i])]:size(2)
    local nWords = train_data[tostring(shuffle[i])]:size(3)

    -- true_batch_size: as prebatched 
    -- batch_size: might change if prebatched is too big
    local true_batch_size = train_data[tostring(shuffle[i])]:size(1)
    local batch_size = true_batch_size

    if opt.debug == 1 then
      io.write('Batch # '..shuffle[i]..' ['..true_batch_size..'x'..nSent..'x'..nWords..'] -> ')
    end

    while batch_size * nSent * nWords > opt.max_batch_vol and batch_size > 5 do
      splits = splits * 2
      batch_size = math.floor(batch_size / 2)
    end

    for s = 1, splits do
      -- could be smaller if this is final split
      local cur_batch_size = batch_size

      -- last batch may be smaller
      if s == splits then
        cur_batch_size = true_batch_size - (splits - 1) * batch_size
      end

      if opt.debug == 1 then
        io.write(' '..cur_batch_size)
      end

      local inputs
      local targets
      if splits == 1 then
        inputs = train_data[tostring(shuffle[i])]
        targets = train_labels[tostring(shuffle[i])]
      else
        inputs = train_data[tostring(shuffle[i])]:narrow(1, 1 + batch_size * (s - 1), cur_batch_size)
        targets = train_labels[tostring(shuffle[i])]:narrow(1, 1 + batch_size * (s - 1), cur_batch_size)
      end

      if opt.cudnn == 1 then
        inputs = inputs:cuda()
        targets = targets:cuda()
      else
        inputs = inputs:double()
        targets = targets:double()
      end

      -- closure to return err, df/dx
      local func = function(x)
        -- get new parameters
        if x ~= params then
          params:copy(x)
        end
        -- reset gradients
        grads:zero()

        -- forward pass
        local outputs = model:forward(inputs)
        local err = criterion:forward(outputs, targets)

        -- track errors and confusion
        total_err = total_err + err * cur_batch_size
        for j = 1, cur_batch_size do
          confusion:add(outputs[j], targets[j])
        end

        -- compute gradients
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)

        if opt.model_type == 'static' then
          -- don't update embeddings for static model
          layers.w2v.gradWeight:zero()
        end

        model:clearState()
        if i % opt.cgfreq == 0 and s == 1 then collectgarbage() end

        return err, grads
      end

      -- gradient descent
      optim_method(func, params, config, state)

      -- reset padding embedding to zero
      layers.w2v.weight[1]:zero()

      -- Renorm (Euclidean projection to L2 ball)
      local renorm = function(row)
        local n = row:norm()
        row:mul(opt.L2s):div(1e-7 + n)
      end

      -- renormalize linear row weights
      for l=1,#layers.linear do
        local w = layers.linear[l].weight
        for j = 1, w:size(1) do
          renorm(w[j])
        end
      end
    end

    if opt.debug == 1 then
      io.write('\n')
    end
  end

  if opt.debug == 1 then
    print('Total err: ' .. total_err / train_size)
    print(confusion)
  end

  -- time taken
  time = timer:time().real - time
  time = opt.batch_size * time / train_size
  if opt.debug == 1 then
    print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
    print(' ')
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end


function Trainer:test(test_data, test_labels, model, criterion)
  model:evaluate()

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local test_size = 0
  local num_batches = 0
  for key, batch in pairs(test_data) do
    num_batches = num_batches + 1
    test_size = test_size + batch:size(1)
  end

  local total_err = 0

  for i=1,num_batches do
    local splits = 1
    local nSent = test_data[tostring(i)]:size(2)
    local nWords = test_data[tostring(i)]:size(3)

    -- true_batch_size: as prebatched 
    -- batch_size: might change if prebatched is too big
    local true_batch_size = test_data[tostring(i)]:size(1)
    local batch_size = true_batch_size

    while batch_size * nSent * nWords > opt.max_batch_vol and batch_size > 5 do
      splits = splits * 2
      batch_size = math.floor(batch_size / 2)
    end

    for s = 1, splits do
      -- could be smaller if this is final split
      local cur_batch_size = batch_size

      -- last batch may be smaller
      if s == splits then
        cur_batch_size = true_batch_size - (splits - 1) * batch_size
      end

      local inputs
      local targets
      if splits == 1 then
        inputs = test_data[tostring(i)]
        targets = test_labels[tostring(i)]
      else
        inputs = test_data[tostring(i)]:narrow(1, 1 + batch_size * (s - 1), cur_batch_size)
        targets = test_labels[tostring(i)]:narrow(1, 1 + batch_size * (s - 1), cur_batch_size)
      end

      if opt.cudnn == 1 then
        inputs = inputs:cuda()
        targets = targets:cuda()
      else
        inputs = inputs:double()
        targets = targets:double()
      end

      local outputs = model:forward(inputs)
      local err = criterion:forward(outputs, targets)

      -- track errors and confusion
      total_err = total_err + err * cur_batch_size
      for j = 1, cur_batch_size do
        confusion:add(outputs[j], targets[j])
      end

      model:clearState()
      if i % opt.cgfreq == 0 and s == 1 then collectgarbage() end
    end
  end

  if opt.debug == 1 then
    print(confusion)
    print('Total err: ' .. total_err / test_size)
    print(' ')
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end

return Trainer
