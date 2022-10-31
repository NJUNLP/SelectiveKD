from comet import download_model, load_from_checkpoint
import sys
import numpy as np

with open(sys.argv[1]) as f:
    src = f.readlines()
with open(sys.argv[2]) as f:
    tgt = f.readlines()
with open(sys.argv[3]) as f:
    ref = f.readlines()

data = []
assert len(src) == len(tgt) == len(ref)
for i in range(len(src)):
    data.append({
        "src": src[i].strip(),
        "mt":  tgt[i].strip(),
        "ref": ref[i].strip()
    })

model_path = download_model("wmt20-comet-da")
model = load_from_checkpoint(model_path)

seg_scores, sys_score = model.predict(data, batch_size=16, gpus=1)

print(sys_score)
print(min(seg_scores), max(seg_scores), np.mean(seg_scores))