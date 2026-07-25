# Sample Images – TrashNet CLI Test

Thư mục này chứa ảnh test tĩnh từ **dataset TrashNet thật** để chạy inference  
qua CLI **không cần camera** (chế độ `CLI_ONLY_INFERENCE`).

## Định dạng

- **Nguồn gốc:** Dataset TrashNet (512×384 JPEG) → resize → raw binary
- **Kích thước:** 224 × 224 pixels (khớp với `kNumCols` × `kNumRows` trong `model_settings.h`)
- **Màu:** RGB888, raw binary (không có header)
- **Dung lượng mỗi file:** 224 × 224 × 3 = **150,528 bytes**

## Nội dung

| File    | Class      | Nguồn gốc               |
|---------|------------|-------------------------|
| image0  | Cardboard  | cardboard1.jpg          |
| image1  | Glass      | glass1.jpg              |
| image2  | Metal      | metal1.jpg              |
| image3  | Paper      | paper1.jpg              |
| image4  | Plastic    | plastic1.jpg            |
| image5  | Trash      | trash1.jpg              |
| image6  | Cardboard  | cardboard10.jpg         |
| image7  | Glass      | glass10.jpg             |
| image8  | Metal      | metal10.jpg             |
| image9  | Plastic    | plastic10.jpg           |

## Tái tạo từ dataset

Dataset TrashNet nằm tại `models/training/dataset-resized/dataset-resized/` trong dự án.  
Nếu chưa có, giải nén từ `models/training/dataset-resized.zip`.

```bash
# Giải nén dataset (nếu cần)
cd models/training
unzip dataset-resized.zip

# Tái tạo sample images từ dataset thật
python3 << 'EOF'
from PIL import Image
import os

# Path tương đối từ thư mục gốc dự án
DATASET = "models/training/dataset-resized/dataset-resized"
OUT     = "static_images/sample_images"
W, H    = 224, 224

mapping = [
    ("cardboard", 0, "image0"), ("glass",     0, "image1"),
    ("metal",     0, "image2"), ("paper",     0, "image3"),
    ("plastic",   0, "image4"), ("trash",     0, "image5"),
    ("cardboard", 1, "image6"), ("glass",     1, "image7"),
    ("metal",     1, "image8"), ("plastic",   1, "image9"),
]

for cls, idx, outname in mapping:
    files = sorted(os.listdir(os.path.join(DATASET, cls)))
    img   = Image.open(os.path.join(DATASET, cls, files[idx])).convert("RGB").resize((W, H))
    with open(os.path.join(OUT, outname), "wb") as f:
        f.write(img.tobytes())
    print(f"✓ {outname} ← {cls}/{files[idx]}")
EOF
```

## Xem ảnh bằng ffmpeg

```bash
ffmpeg -f rawvideo -pixel_format rgb24 -video_size 224x224 \
       -i static_images/sample_images/image0 image0_cardboard.bmp
```

## Thêm ảnh của bạn

```python
from PIL import Image

def to_raw(src_jpg, dst_raw):
    img = Image.open(src_jpg).convert("RGB").resize((224, 224))
    with open(dst_raw, "wb") as f:
        f.write(img.tobytes())
    print(f"✓ {dst_raw}  ({224*224*3:,} bytes)")

# Ví dụ:
to_raw("my_photo.jpg", "static_images/sample_images/image0")
```
