local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'nngraph'

require 'nn'
require 'image'
require 'nn'

require 'cutorch'
require 'cunn'
require 'cudnn'


opt = lapp[[
     --input_height  (default 512)  frame_height, must be a multiple of 64
     --model   (default 'FlowNetS_SmallDisp.t7')
     --video   (default '')
     --output_height  (default 512) height of output flowmap
     --shift   (default 2) temporal distance between 2 frames
     --verbose
     ]]



local cap = cv.VideoCapture{opt.video_path ~= '' and opt.video_path or 0}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

cv.namedWindow{opt.network, cv.WINDOW_AUTOSIZE}
local _,frame = cap:read{}

frame1 = nil
frame2 = nil
frames_array = {}
frames = nil
local function preprocess(frame)
   frame2 = frames_array[1]
   frame1 = frame:permute(3,2,1):float() / 255
   frame1 = image.crop(frame1,'c',frame1:size(3),frame1:size(3)*1.3333333)
   frame1 = image.scale(frame1, opt.input_height)
   
   table.insert(frames_array,frame1)
   if frame2 then
    frames = torch.cat(frame1,frame2,1)
   end
   if #frames_array == opt.shift then
    table.remove(frames_array,1)
   end
   return frames
end

local net = torch.load(opt.model):cuda()
net:evaluate()
net = cudnn.convert(net,cudnn) --will trigger a warning for graph but it still works


preprocess(frame)
preprocess(frame)
local input = frames:cuda()
if opt.verbose then
  print({input})
  print({net:forward(input)}) --to check what the output of the network is like
end

while true do
    local input = preprocess(frame):cuda()
    
    local output = net:forward(input)
    if torch.type(output) == 'table' then
     output = output[5]:float()
    else
     output = output:float()
    end
    local output_ = torch.FloatTensor(3,output:size(2),output:size(3))
    if opt.verbose then
      print(output:max())
      print(output:min())
    end
    output_[1]:fill(255)
    output_[{{2,3}}]:copy(100*(512/opt.input_height)*output)
    local out = image.scale(image.yuv2rgb(output_),opt.output_height)
    cv.imshow{opt.model, torch.clamp(out:permute(3,2,1),0,255):byte()}
    if cv.waitKey{1} >= 0 then break end
    cap:read{frame}
end
