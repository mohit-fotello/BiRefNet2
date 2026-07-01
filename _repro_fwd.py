import os, sys, time, signal, torch, torch.nn as nn

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
from config import Config
from models.birefnet import BiRefNet

def alarm(sig, frm):
    print("[TIMEOUT]", flush=True); os._exit(7)
signal.signal(signal.SIGALRM, alarm)
signal.alarm(180)

torch.set_float32_matmul_precision('high')
m = BiRefNet(bb_pretrained=False).cuda().train()
x = torch.randn(bs, 3, 1024, 1024, device='cuda')
print(f"mode=single+backward bs={bs}", flush=True)
for it in range(3):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = m(x)
        preds = out[0]
        if isinstance(preds, (list, tuple)) and isinstance(preds[0], (list, tuple)):
            preds = preds[1]
        loss = sum(p.float().mean() for p in preds)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"iter {it}: {time.time()-t:.3f}s peak={peak:.2f}GB", flush=True)
print("OK", flush=True)
