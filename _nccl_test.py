import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
t = torch.ones(1024, 1024, device=f"cuda:{lr}")
dist.all_reduce(t)
torch.cuda.synchronize()
if lr == 0:
    print("ALLREDUCE_OK sum=", t.sum().item(), flush=True)
dist.destroy_process_group()
