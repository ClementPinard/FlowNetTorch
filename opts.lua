--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-overWrite', false, 'overwrite cache files (when the dataset has changed')
    cmd:option('-data', './data', 'Home of flyingchairs dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-cudnnMode',          1, 'mode for cudnn, 0: normal, 1 (default) : benchmark, 2: fastest')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-width',         512,    'image width')
    cmd:option('-height',        384,    'image height')
    cmd:option('-cropWith',      448,    'random crop width for training')
    cmd:option('-cropHeight',    320,    'random crop height for training')
    cmd:option('-downSample',      4,    'downSample of network (will be overwritten if specified in model definition)')
    cmd:option('-scales',          5,    'number of scales for multiscale loss (overwritten by model definition)')
    cmd:option('-noColor',         false,    'only work on Y chanel of the image')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       0, 'Number of batches per epoch; if not set, will match Dataset size')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    1e-4, 'learning rate; see train.lua for decay recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'FlowNetS', 'Options: More to come after')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-loss',        'EPE',  'loss function for flowmaps (EPE, Abs, MSE or SmoothL1)')
    ----------Log options ------------------------------------
    cmd:option('-logAllScales',  false, 'log all scales used for training (can be messy in the final graph')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true, overWrite=true}))
    -- add date/time
     --opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M
