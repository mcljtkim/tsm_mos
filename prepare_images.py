"""
prepare_images.py
=================
Generate study images (Original / TSM-HB / Forbes-HB) from ImageNet validation set
and write study_config.json for the GitHub Pages user study.

Directory layout produced:
  images/
    Original/<class>/<idx>.jpg
    TSM/<class>/<idx>.jpg
    Forbes/<class>/<idx>.jpg    (optional, if --methods includes Forbes)
  study_config.json

Usage (run from the user_study_imagenet directory):
  cd /hdd_ext/hdd2000/ECCV2026/result/rebuttal/user_study_imagenet
  CUDA_VISIBLE_DEVICES=0 python prepare_images.py \
      --imagenet_root /hdd_ext/hdd2000/dataset/ImageClassification/ImageNet \
      --n_per_class 8 \
      --methods Original TSM Forbes \
      --out_size 256 \
      --gpu 0

After running, commit the entire user_study_imagenet/ directory to GitHub
and enable GitHub Pages (Settings → Pages → Deploy from branch → root).
"""

import sys, os, argparse, json, time, shutil

# BRS import: code root is 3 levels up from result/rebuttal/user_study_imagenet/
CODE_ROOT = '/mnt/c12606b4-b83f-4123-8af2-a606e90b10fe/CVPR2027/code/v1'
sys.path.insert(0, CODE_ROOT)

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import timm

# ── 10 study classes (broad categories → all ImageNet-1K folder indices) ──────
# Folder names in this dataset are zero-padded 5-digit class indices (00000–00999).
# Each broad class aggregates ALL fine-grained ImageNet synsets that belong to it,
# so participants see "cat" not "tabby vs Persian", matching the CIFAR-10 setup.
STUDY_CLASSES = [
    ("cat",      ["%05d" % i for i in range(281, 286)]),            # 281-285: all domestic cat breeds
    ("dog",      ["%05d" % i for i in list(range(151, 276))]),      # 151-275: all dog breeds (125 synsets)
    ("bird",     ["%05d" % i for i in list(range(7, 25))]),         # 7-24: all bird species
    ("car",      ["%05d" % i for i in [436, 511, 627, 661, 751, 817, 829, 867]]),  # cars & trucks
    ("airplane", ["%05d" % i for i in [404, 895]]),                 # airliner, warplane
    ("ship",     ["%05d" % i for i in [484, 510, 554, 625, 628, 724, 780, 814, 871]]),  # boats & ships
    ("horse",    ["%05d" % i for i in [339, 603]]),                 # sorrel horse, horse cart
    ("elephant", ["%05d" % i for i in [385, 386]]),                 # Indian & African elephant
    ("flower",   ["%05d" % i for i in [985, 986, 987]]),            # daisy, orchid, sunflower
    ("bicycle",  ["%05d" % i for i in [444, 671, 665]]),            # bicycle, mountain bike, moped
]
CLASS_NAMES = [c[0] for c in STUDY_CLASSES]

# Forbes is imported from the model package (see below)


def tensor_to_pil(x, mean, std, out_size):
    """Denormalize ImageNet tensor and return PIL image at out_size."""
    dev = x.device
    x_01 = x.squeeze(0).cpu() * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
    x_01 = x_01.clamp(0, 1)
    pil = T.ToPILImage()(x_01)
    return pil.resize((out_size, out_size), Image.LANCZOS)


def find_imagenet_images(imagenet_root, folder_ids, n):
    """Find up to n image paths from ImageNet for given folder IDs.
    Folder IDs are zero-padded 5-digit class indices (e.g. '00281' for tabby cat).
    """
    paths = []
    for fid in folder_ids:
        cls_dir = os.path.join(imagenet_root, fid)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted([
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        paths.extend(imgs)
        if len(paths) >= n:
            break
    return paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet_root', default='/hdd_ext/hdd2000/dataset/ImageClassification/ImageNet')
    ap.add_argument('--n_per_class',   type=int, default=8,
                    help='Images per class per condition (total = n_per_class × 10 × n_conditions)')
    ap.add_argument('--methods',       nargs='+', default=['Original', 'TSM', 'Forbes'],
                    choices=['Original', 'TSM', 'Forbes'])
    ap.add_argument('--trials_per_condition', type=int, default=None,
                    help='Max images per condition shown to each participant (default: n_per_class×10)')
    ap.add_argument('--out_size',      type=int, default=256)
    ap.add_argument('--gpu',           type=int, default=0)
    ap.add_argument('--out_dir',       default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu) if torch.cuda.is_available() else None
    img_dir = os.path.join(args.out_dir, 'images')

    MEAN = [0.485, 0.456, 0.406];  STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    # Load model
    need_model = any(m in args.methods for m in ['TSM', 'Forbes'])
    model = None
    if need_model:
        print("Loading ResNet-50 ...")
        model = timm.create_model('resnet50', pretrained=True).to(device)
        for p in model.parameters(): p.requires_grad_(False)
        model.eval()

    # Load BRS for TSM
    brs = None
    if 'TSM' in args.methods:
        try:
            from model import BRS_0218_v4
            brs = BRS_0218_v4(model, device)
            print("BRS_0218_v4 loaded for TSM-HB.")
        except ImportError:
            print("WARNING: could not import BRS_0218_v4. TSM will be skipped.")
            args.methods = [m for m in args.methods if m != 'TSM']

    # Load Forbes for Forbes-HB baseline
    # Use 16×16 parameter grid so pixel blocks are 224/16 = 14px (matching paper setting).
    # Default Forbes_224 has a 14×14 grid → 16px blocks, which is too coarse.
    forbes = None
    if 'Forbes' in args.methods:
        try:
            from model.InputCorrectionTorch import Forbes_224 as _Forbes_224

            class Forbes_224_B14(_Forbes_224):
                """Forbes_224 with 16×16 grid → 14px pixel blocks."""
                def __init__(self, model, device):
                    super().__init__(model, device)
                    self.base_transform_size = {
                        'Warping_224':         [1, 2, 16, 16],
                        'Scaling_224':         [16, 16, 1, 3, 1, 1],
                        'Sinusoid_224':        [16, 16, 3, 1, 1],
                        'Speckle_224':         [1, 3, 16, 16],
                        'Averaging_blend_224': [3, 3, 16, 16],
                        'Noising_blend_224':   [3, 3, 16, 16],
                    }

                def _size_init(self):
                    b, c, h, w = self.in_img.shape
                    self.transform_size = {
                        'Warping_224':         [b, 2, 16, 16],
                        'Scaling_224':         [16, 16, b, 3, 1, 1],
                        'Sinusoid_224':        [16, 16, b, 3, 1, 1],
                        'Speckle_224':         [b, 3, 16, 16],
                        'Averaging_blend_224': [3, b, 3, 16, 16],
                        'Noising_blend_224':   [3, b, 3, 16, 16],
                    }

            forbes = Forbes_224_B14(model, device)
            print("Forbes_224_B14 loaded (16×16 grid → 14px pixel blocks).")
        except ImportError as e:
            print(f"WARNING: could not import Forbes_224: {e}. Forbes will be skipped.")
            args.methods = [m for m in args.methods if m != 'Forbes']

    all_entries = []
    t0 = time.time()

    for cls_idx, (cls_name, synsets) in enumerate(STUDY_CLASSES):
        print(f"\n[{cls_idx+1}/10] Class: {cls_name}")
        paths = find_imagenet_images(args.imagenet_root, synsets, args.n_per_class)
        if not paths:
            print(f"  WARNING: no images found for {cls_name} ({synsets})")
            continue
        print(f"  Found {len(paths)} images")

        for img_idx, img_path in enumerate(paths):
            try:
                pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  skip {img_path}: {e}"); continue

            x = transform(pil).unsqueeze(0).to(device)

            for method in args.methods:
                out_subdir = os.path.join(img_dir, method, cls_name)
                os.makedirs(out_subdir, exist_ok=True)
                out_path = os.path.join(out_subdir, f'{img_idx}.jpg')

                if os.path.exists(out_path):
                    print(f"  [{method}] {cls_name}/{img_idx}.jpg already exists, skip")
                else:
                    if method == 'Original':
                        result_pil = tensor_to_pil(x, MEAN, STD, args.out_size)

                    elif method == 'TSM':
                        adv, *_ = brs.optimize(
                            x, md_sign=+1.0, hd_sign=-1.0,
                            rho_target=0.2, stop_md_loss=0.10,
                            lr_s=1.0, lr_p=0.01,
                            clamp_range=(-3.0, 3.0),
                        )
                        result_pil = tensor_to_pil(adv, MEAN, STD, args.out_size)

                    elif method == 'Forbes':
                        adv = forbes.optimize(x)
                        result_pil = tensor_to_pil(adv, MEAN, STD, args.out_size)

                    result_pil.save(out_path, quality=92)
                    print(f"  [{method}] {cls_name}/{img_idx}.jpg saved", end='\r')

                # Relative path from out_dir (for study_config.json)
                rel = os.path.relpath(out_path, args.out_dir).replace('\\', '/')
                all_entries.append({
                    "path":      rel,
                    "condition": method,
                    "class_id":  cls_idx,
                    "class_name": cls_name,
                })

    print(f"\n\nAll images generated in {(time.time()-t0)/60:.1f} min")

    # Write study_config.json
    tpc = args.trials_per_condition or (args.n_per_class * len(STUDY_CLASSES))
    config = {
        "classes":               CLASS_NAMES,
        "conditions":            args.methods,
        "trials_per_condition":  tpc,
        "n_per_class":           args.n_per_class,
        "contact_email":         "victoryjinta@gmail.com",
        "images":                all_entries,
    }
    cfg_path = os.path.join(args.out_dir, 'study_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"study_config.json written ({len(all_entries)} entries)  →  {cfg_path}")
    print(f"\nNext steps:")
    print(f"  1. Commit user_study_imagenet/ to GitHub")
    print(f"  2. Enable GitHub Pages (Settings → Pages → Deploy from branch → root or /docs)")
    print(f"  3. Share the Pages URL with participants")


if __name__ == '__main__':
    main()
