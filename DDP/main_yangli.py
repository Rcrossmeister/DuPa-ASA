################
## main.py文件
import torch
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 新增：从外面得到local_rank参数
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# 新增：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
# load模型要在构造DDP模型之前，且只需要在master上加载就行了。
if dist.get_rank() == 0:
    model.load_state_dict(torch.load(ckpt_path))
# 新增：构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 优化器：要在构造DDP model之后，才能初始化model。
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 构造数据
my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
# 新增：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
#       sampler的原理，后面也会介绍。
train_sampler = torch.distributed.DistributedSampler(my_trainset)
# 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size,
                                          sampler=train_sampler)

# 网络训练
model.train()
for epoch in range(num_epochs):
    # 新增：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)