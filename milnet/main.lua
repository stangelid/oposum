require 'hdf5'
require 'nn'
require 'optim'
require 'lfs'

-- Flags
cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Multiple Instance Larning Newtork')
cmd:text()
cmd:text('Options')
cmd:option('-emb_type', 'nonstatic', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings, static during learning), nonstatic (pre-trained embeddings, tuned during learning)')
cmd:option('-model', 'milnet', 'milnet/hiernet')
cmd:option('-data', '', 'Training, development, test and word2vec data in .hdf5 format')
cmd:option('-seg_data', '', 'Selected batches for segment-level predictions')
cmd:option('-seg_info', '', 'Segment-level info for above selected batches (codes, labels, original text)')
cmd:option('-seg_out', '', 'File to store predicted polarities')
cmd:option('-cudnn', 1, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
cmd:option('-folds', 3, 'Number of folds to use. If test set provided, folds=1. max 10')
cmd:option('-force_cv', 1, 'force CV even if data have dev/test splits')
cmd:option('-debug', 0, 'print debugging info including timing, confusions')
cmd:option('-subset', 0, 'only load a subset of batches (used for debugging)')
cmd:option('-gpuid', 1, 'GPU device id to use.')
cmd:option('-savefile', '', 'Name of output file, which will hold the trained model, model parameters, and training scores. Default filename is TIMESTAMP_results')
cmd:option('-savemodel', 0, 'save learned model parameters. Default is 0.')
cmd:option('-zero_indexing', 1, 'If data is zero indexed')
cmd:option('-seed', 1, 'random seed, set -1 for actual random')
cmd:text()

-- Training own dataset
cmd:option('-train_only', 0, 'Set to 1 to only train on data. Default is cross-validation')
cmd:option('-test_only', 0, 'Set to 1 to only do testing. Must have a -warm_start_model')
cmd:option('-warm_start_model', '', 'Path to .t7 file with pre-trained model. Should contain a table with key \'model\'')
cmd:option('-batch_size', 50, 'Batch size for training')
cmd:text()

-- Training hyperparameters
cmd:option('-num_epochs', 25, 'Number of training epochs')
cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
cmd:option('-L2s', 3, 'L2 normalize weights')
cmd:option('-max_batch_vol', 400000, 'If nInstances x nSentences x nWords greater than this, split in half')
cmd:option('-force_max_sent', 5000, 'set maximum sentence length')
cmd:option('-force_max_doc', 500, 'set maximum document length (in sentences, EDUs etc.)')
cmd:option('-cgfreq', 25, 'Collect garbage every cgfreq batches')
cmd:text()

-- Model hyperparameters
cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
cmd:option('-num_doc_feat_maps', 100, 'Number of feature maps after 2nd convolution')
cmd:option('-kernels', '{3,4,5}', 'Kernel sizes of 1st convolutions, table format.')
cmd:option('-dropout_p', 0.5, 'p for dropout')
cmd:option('-pool', 'max', 'type of pooling if no attention used (max/mean)')
cmd:option('-doc_rnn_hiddensize', 150, 'size of gru/lstm hidden vectors of document encoder')
cmd:option('-doc_gru_dropout', 0.5, 'p for internal gru dropout')
cmd:option('-doc_rho', 10, 'back-propagation through time steps')
cmd:option('-doc_att', 0, 'use attention on document encoder')
cmd:option('-doc_att_size', 0, 'size of attention vector on document level')
cmd:option('-doc_att_dropout', -1, 'dropout on attention (-1 to disable)')
cmd:text()

function get_layer(model, name)
  local named_layer
  if name == 'linear' then
    named_layer = {}
  end

  function get(layer)
    if torch.typename(layer) == name or layer.name == name then
      if name == 'linear' then
        table.insert(named_layer, layer)
      else
        named_layer = layer
      end
    end
  end

  model:apply(get)
  return named_layer
end

function get_class_weights(nClasses)
  local w = torch.Tensor(nClasses)
  for i=1,nClasses do
    w[i] = i
  end

  w = w - (nClasses + 1)/2
  w = 2 * w / (nClasses - 1)

  return w
end

-- build model for training
function build_model(w2v)
  local ModelBuilder
  if opt.model == 'hiernet' then
    ModelBuilder = require 'model.hierAtt'
  elseif opt.model == 'milnet' then
    ModelBuilder = require 'model.hierMIL'
  end
  local model_builder = ModelBuilder.new(w2v)

  local model

  if opt.warm_start_model == '' then
    model = model_builder:make_net()
  else
    require "nngraph"
    if opt.cudnn == 1 then
      require "cudnn"
      require "cunn"
    end
    model = torch.load(opt.warm_start_model).model
  end

  local criterion = nn.ClassNLLCriterion()

  -- move to GPU
  if opt.cudnn == 1 then
    model = model:cuda()
    criterion = criterion:cuda()
  end

  -- get layers
  local layers = {}
  layers['linear'] = get_layer(model, 'linear')
  layers['w2v'] = get_layer(model, 'nn.LookupTable')

  return model, criterion, layers
end

function train_loop(all_train, all_train_label, test, test_label, dev, dev_label, w2v)
  -- Initialize objects
  local Trainer = require 'trainer'
  local trainer = Trainer.new()

  local optim_method
  if opt.optim_method == 'adadelta' then
    optim_method = optim.adadelta
  elseif opt.optim_method == 'adam' then
    optim_method = optim.adam
  end

  local best_model -- save best model
  local fold_dev_scores = {}
  local fold_test_scores = {}

  local train = {}
  local train_label = {}
  local test = test or {}
  local test_label = test_label or {}
  local dev = dev or {}
  local dev_label = dev_label or {}

  local num_batches = 0
  for key, batch in pairs(all_train) do
    num_batches = num_batches + 1
  end

  local shuffle = torch.randperm(num_batches)

  -- Training folds.
  for fold = 1, opt.folds do
    local timer = torch.Timer()
    local fold_time = timer:time().real

    print()
    print('==> fold ', fold)

    opt.batch_size = all_train['1']:size(1)

    if opt.has_test == 0 and opt.train_only == 0 then
      local i_start = math.floor((fold - 1) * (num_batches / opt.folds) + 1)
      local i_end = math.floor(fold * (num_batches / opt.folds))
      for i=1,i_end-i_start+1 do
        test[tostring(i)] = all_train[tostring(shuffle[i+i_start-1])]
        test_label[tostring(i)] = all_train_label[tostring(shuffle[i+i_start-1])]
      end
      local i_actual = 1
      for i=1,num_batches do
        if i < i_start or i > i_end then
          train[tostring(i_actual)] = all_train[tostring(shuffle[i])]
          train_label[tostring(i_actual)] = all_train_label[tostring(shuffle[i])]
          i_actual = i_actual + 1
        end
      end
    else
      train = all_train
      train_label = all_train_label
    end

    if opt.has_dev == 0 then
      -- shuffle train to get dev/train split (10% to dev)
      -- We organize our data in batches at this split before epoch training.
      local num_full_train_batches = 0
      for key, batch in pairs(train) do
        num_full_train_batches = num_full_train_batches + 1
      end
      local shuffle_train = torch.randperm(num_full_train_batches):add(-opt.zero_indexing)

      local num_train_batches = torch.round(num_full_train_batches * 0.9)
      local num_dev_batches = num_full_train_batches - num_train_batches
      
      local new_train = {}
      local new_train_label = {}
      for i=1,num_train_batches do
        key = tostring(shuffle_train[i])
        new_train[tostring(i)] = train[key]
        new_train_label[tostring(i)] = train_label[key]
      end
      for i=1,num_dev_batches do
        key = tostring(shuffle_train[i + num_train_batches])
        dev[tostring(i)] = train[key]
        dev_label[tostring(i)] = train_label[key]
      end

      train = new_train
      train_label = new_train_label
    end
    print('data ready..')

    if opt.doc_att == 1 and opt.doc_att_size == 0 then
      opt.doc_att_size = 2*opt.doc_rnn_hiddensize
    end

    local model, criterion, layers = build_model(w2v)
    print('model ready..')

    -- Call getParameters once
    local params, grads = model:getParameters()

    -- Training loop.
    best_model = model:clone()
    local best_epoch = 1
    local best_err = 0.0

    -- Training.
    -- Gradient descent state should persist over epochs
    local state = {}
    for epoch = 1, opt.num_epochs do
      local epoch_time = timer:time().real

      -- Train
      local train_err = trainer:train(train, train_label, model, criterion, optim_method, layers, state, params, grads)
      -- Dev
      local dev_err = trainer:test(dev, dev_label, model, criterion, layers)

      if dev_err > best_err then
        best_model = model:clone()
        best_epoch = epoch
        best_err = dev_err 
      end

      if opt.debug == 1 then
        print()
        print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
        print('\n')
      end

      print('epoch:', epoch, 'train perf:', 100*train_err, '%, val perf ', 100*dev_err, '%')
    end

    print('best dev err:', 100*best_err, '%, epoch ', best_epoch)
    table.insert(fold_dev_scores, best_err)

    -- Testing.
    if opt.train_only == 0 then
      local test_err = trainer:test(test, test_label, best_model, criterion, layers)
      print('test perf ', 100*test_err, '%')
      table.insert(fold_test_scores, test_err)
    end

    if opt.debug == 1 then
      print()
      print('time for one fold: ', (timer:time().real - fold_time * 1000), 'ms')
      print('\n')
    end
  end

  return fold_dev_scores, fold_test_scores, best_model
end

-- writes MILNET predictions to output file
function output_milnet(fname, model, seg_data, seg_codes, seg_docids, seg_segids, seg_orig)
  model:evaluate()
  local par_cl = get_layer(model, 'par_cl')
  local att = get_layer(model, 'concat_att').modules[1]
  local classW = get_class_weights(opt.num_classes):cuda()

  local num_batches = 0
  for key, batch in pairs(seg_data) do
    num_batches = num_batches + 1
  end

  local fmil = io.open(fname, 'w')
  for i=1,num_batches do
    local key = tostring(i-1)
    local batch_size = seg_data[key]:size(1)
    local inputs = seg_data[key]
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
    else
      inputs = inputs:double()
    end

    local outputs = model:forward(inputs)
    local seg_preds = par_cl.output
    local seg_att = att.output

    for j=1,batch_size do
      local polarities = torch.mv(seg_preds[j], classW)
      local gated_polarities = torch.cmul(polarities, seg_att[j])

      for k=1,#seg_segids[i][j] do
          fmil:write(string.format("%s\t%.10f\t%.10f\t%.10f\t%s\n", seg_codes[i][j][k], polarities[k], seg_att[j][k], gated_polarities[k], seg_orig[i][j][k]))
      end
    end
  end
  fmil:close()
end

-- writes HIERNET predictions to file
function output_hiernet(fname, model, seg_data, seg_codes, seg_docids, seg_segids, seg_orig)
  model:evaluate()
  local att = get_layer(model, 'concat_att').modules[1]
  local classW = get_class_weights(opt.num_classes):cuda()

  local num_batches = 0
  for key, batch in pairs(seg_data) do
    num_batches = num_batches + 1
  end

  local fmil = io.open(fname, 'w')
  for i=1,num_batches do
    local key = tostring(i-1)
    local batch_size = seg_data[key]:size(1)
    local num_segs = seg_data[key]:size(2)
    local inputs = seg_data[key]
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
    else
      inputs = inputs:double()
    end

    local outputs = model:forward(inputs)
    local doc_preds = torch.exp(outputs:view(batch_size, 1, -1):expand(batch_size, num_segs, opt.num_classes))
    local seg_att = att.output

    for j=1,batch_size do
      local polarities = torch.mv(doc_preds[j], classW)
      local gated_polarities = torch.cmul(polarities, seg_att[j])

      for k=1,#seg_segids[i][j] do
          fmil:write(string.format("%s\t%.10f\t%.10f\t%.10f\t%s\n", seg_codes[i][j][k], polarities[k], seg_att[j][k], gated_polarities[k], seg_orig[i][j][k]))
      end
    end
  end
  fmil:close()
end

-- loads data from .hdf5 file
function load_data()
  local train, train_label
  local dev, dev_label
  local test, test_label

  print('loading data...')
  local f = hdf5.open(opt.data, 'r')

  local w2v = f:read('w2v'):all()

  train = f:read('train'):all()
  train_label = f:read('train_label'):all()
  assert(torch.type(train) == 'table',
    'Data not stored in prebatched mode. Provide correct file')

  if opt.subset > 0 then
    new_train = {}
    new_train_label = {}
    for i=(1-opt.zero_indexing),(opt.subset-opt.zero_indexing) do
      new_train[tostring(i)] = train[tostring(i)]
      new_train_label[tostring(i)] = train_label[tostring(i)]
    end
    train = new_train
    train_label = new_train_label
  end

  opt.num_classes = 0

  for key, batch in pairs(train) do
    if batch:size(2) > opt.force_max_doc then
      train[key] = train[key][{{},{1,opt.force_max_doc},{}}]
    end
    if batch:size(3) > opt.force_max_sent then
      train[key] = train[key][{{},{},{1,opt.force_max_sent}}]
    end
    if opt.zero_indexing == 1 then
      train[key]:add(1)
      train_label[key]:add(1)
    end
  end
  for key, batch in pairs(train_label) do
    opt.num_classes = math.max(opt.num_classes, torch.max(batch))
  end

  if opt.force_cv == 1 then
    opt.has_dev = 0
  else
    opt.has_dev = 1
    dev = f:read('dev'):all()
    dev_label = f:read('dev_label'):all()

    for key, batch in pairs(dev) do
      if batch:size(2) > opt.force_max_doc then
        dev[key] = dev[key][{{},{1,opt.force_max_doc},{}}]
      end
      if batch:size(3) > opt.force_max_sent then
        dev[key] = dev[key][{{},{},{1,opt.force_max_sent}}]
      end
      if opt.zero_indexing == 1 then
        dev[key]:add(1)
        dev_label[key]:add(1)
      end
    end
    for key, batch in pairs(dev_label) do
      opt.num_classes = math.max(opt.num_classes, torch.max(batch))
    end
  end

  if opt.force_cv == 1 then
    opt.has_test = 0
  else
    opt.has_test = 1
    test = f:read('test'):all()
    test_label = f:read('test_label'):all()

    for key, batch in pairs(test) do
      if batch:size(2) > opt.force_max_doc then
        test[key] = test[key][{{},{1,opt.force_max_doc},{}}]
      end
      if batch:size(3) > opt.force_max_sent then
        test[key] = test[key][{{},{},{1,opt.force_max_sent}}]
      end
    end
    if opt.zero_indexing == 1 then
      test[key]:add(1)
      test_label[key]:add(1)
    end
    for key, batch in pairs(test_label) do
      opt.num_classes = math.max(opt.num_classes, torch.max(batch))
    end
  end

  print('data loaded!')
  f:close()

  return train, train_label, test, test_label, dev, dev_label, w2v
end

-- loads data from .hdf5 file
function load_seg_data()
  local seg_data
  local seg_label
  local seg_codes = {}
  local seg_docids = {}
  local seg_segids = {}
  local seg_orig = {}

  print('loading data...')
  local f = hdf5.open(opt.seg_data, 'r')

  seg_data = f:read('data'):all()
  assert(torch.type(seg_data) == 'table',
    'Data not stored in prebatched mode. Provide correct file')
  seg_label = f:read('label'):all()
  f:close()

  if opt.seg_info == '' then
    opt.seg_info = opt.seg_data:sub(1, -6)..'.info'
  end

  for line in io.lines(opt.seg_info) do
    fields = line:split('\t')
    code = fields[1]
    ids = fields[2]:split(' ')
    orig = fields[4]
    batchid = tonumber(ids[1])
    docid = tonumber(ids[2])
    segid = tonumber(ids[3])
    if opt.zero_indexing == 1 then
      batchid = batchid + 1
      docid = docid + 1
      segid = segid + 1
    end

    if seg_codes[batchid] == nil then
      seg_codes[batchid] = {}
      seg_docids[batchid] = {}
      seg_segids[batchid] = {}
      seg_orig[batchid] = {}
    end

    table.insert(seg_docids[batchid], docid)
    if seg_segids[batchid][docid] == nil then
      seg_codes[batchid][docid] = {}
      seg_segids[batchid][docid] = {}
      seg_orig[batchid][docid] = {}
    end
    table.insert(seg_codes[batchid][docid], code)
    table.insert(seg_segids[batchid][docid], segid)
    table.insert(seg_orig[batchid][docid], orig)
  end

  if opt.zero_indexing == 1 then
    for key in pairs(seg_data) do
      seg_data[key]:add(1)
      seg_label[key]:add(1)
    end
  end

  print('segment-level data loaded!')

  return seg_data, seg_label, seg_codes, seg_docids, seg_segids, seg_orig
end

function main()
  -- parse arguments
  opt = cmd:parse(arg)

  if opt.seed ~= -1 then
    torch.manualSeed(opt.seed)
  end
  if opt.cudnn == 1 then
    require 'cutorch'
    if opt.seed ~= -1 then
      -- 'All' means all GPUs
      cutorch.manualSeedAll(opt.seed)
    end
    cutorch.setDevice(opt.gpuid)
  end

  -- Read HDF5 data
  local train, train_label
  local test, test_label
  local dev, dev_label
  local w2v

  train, train_label, test, test_label, dev, dev_label, w2v = load_data()
  seg_data, seg_label, seg_codes, seg_docids, seg_segids, seg_orig = load_seg_data()

  opt.vocab_size = w2v:size(1)
  opt.vec_size = w2v:size(2)
  opt.max_sent = 0
  opt.max_doc = 0
  for key, batch in pairs(train) do
    opt.max_sent = math.max(batch:size(3), opt.max_sent)
    opt.max_doc = math.max(batch:size(2), opt.max_doc)
  end
  if dev then
    for key, batch in pairs(dev) do
      opt.max_sent = math.max(batch:size(3), opt.max_sent)
      opt.max_doc = math.max(batch:size(2), opt.max_doc)
    end
  end
  if test then
    for key, batch in pairs(test) do
      opt.max_sent = math.max(batch:size(3), opt.max_sent)
      opt.max_doc = math.max(batch:size(2), opt.max_doc)
    end
  end

  print('vocab size: ', opt.vocab_size)
  print('vec size: ', opt.vec_size)
  print('max sentence length: ', opt.max_sent)
  print('max document length: ', opt.max_doc)

  loadstring("opt.kernels = " .. opt.kernels)()

  if opt.test_only == 1 then
    assert(opt.warm_start_model ~= '', 'must have -warm_start_model for testing')
    assert(opt.has_test == 1)
    local Trainer = require "trainer"
    local trainer = Trainer.new()
    print('Testing...')
    local model, criterion = build_model(w2v)
    local test_err = trainer:test(test, test_label, model, criterion, layers)
    print('Test score:', test_err)
    os.exit()
  end

  if opt.has_test == 1 or opt.train_only == 1 or opt.seg_data ~= '' then
    -- don't do CV if we have a test set, or are training only
    opt.folds = 1
  end

  if opt.seg_data ~= '' then
    opt.train_only = 1
  end

  -- training loop
  local fold_dev_scores, fold_test_scores, best_model = train_loop(train, train_label, test, test_label, dev, dev_label, w2v)

  if opt.model == 'milnet' then
    print('Producing MILNET output')
    output_milnet(opt.seg_out, best_model, seg_data, seg_codes, seg_docids, seg_segids, seg_orig)
  else
    print('Producing HIERNET output')
    output_hiernet(opt.seg_out, best_model, seg_data, seg_codes, seg_docids, seg_segids, seg_orig)
  end

  print('dev scores:')
  print(fold_dev_scores)
  print('average dev score: ', torch.Tensor(fold_dev_scores):mean())

  if opt.train_only == 0 then
    print('test scores:')
    print(fold_test_scores)
    print('average test score: ', torch.Tensor(fold_test_scores):mean())
  end

  -- make sure output directory exists
  if not path.exists('results') then lfs.mkdir('results') end

  local savefile
  if opt.savefile ~= '' then
    savefile = opt.savefile
  else
    savefile = string.format('results/%s_model.t7', os.date('%Y%m%d_%H%M'))
  end
  print('saving results to ', savefile)

  local save = {}
  save['dev_scores'] = fold_dev_scores
  if opt.train_only == 0 then
    save['test_scores'] = fold_test_scores
  end
  save['opt'] = opt
  if opt.savemodel > 0 then
    save['model'] = best_model
    save['embeddings'] = get_layer(best_model, 'nn.LookupTable').weight
  end
  torch.save(savefile, save)
end

main()
