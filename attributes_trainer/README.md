## Additional repository for network learning on face / gender recognition tasks for ChaLearn Face Anti-spoofing Attack Detection Challenge

### Step 1
Download AFAD-Lite and unpack in ```data/afad-lite```

### Step 2
Run train on 4 GPU's
```CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config data/opts/```

### Step 3
Use trained model in main repository for ```exp2```

