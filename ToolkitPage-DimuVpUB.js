import{r as a,j as e}from"./index-CkE3sGvL.js";const o=[{name:"pytorch/pytorch",url:"https://github.com/pytorch/pytorch",purpose:"Framework chính cho deep learning",note:"Được phép trong đề thi IOAI (theo Technical Appendix)",cat:"Framework"},{name:"huggingface/transformers",url:"https://github.com/huggingface/transformers",purpose:"NLP, audio, multimodal, fine-tune",note:"Thi có thể không cho tải model ngoài — chỉ dùng model cài sẵn",cat:"Framework"},{name:"huggingface/pytorch-image-models",url:"https://github.com/huggingface/pytorch-image-models",purpose:"Hàng trăm model ảnh pretrained (timm)",note:"Cực tiện cho mọi bài CV; nhớ API create_model",cat:"CV"},{name:"ultralytics/ultralytics",url:"https://github.com/ultralytics/ultralytics",purpose:"YOLO detection & segmentation",note:"License AGPL — chỉ học tập/thi, không thương mại hoá",cat:"CV"},{name:"facebookresearch/detr",url:"https://github.com/facebookresearch/detr",purpose:"Object detection bằng transformer",note:"Đọc hiểu kiến trúc DETR",cat:"CV"},{name:"milesial/Pytorch-UNet",url:"https://github.com/milesial/Pytorch-UNet",purpose:"Segmentation baseline gọn",note:"Copy cấu trúc U-Net, sửa theo đề",cat:"CV"},{name:"openai/CLIP",url:"https://github.com/openai/CLIP",purpose:"Vision-language, zero-shot",note:"Học embedding ảnh–text",cat:"CV"},{name:"openai/whisper",url:"https://github.com/openai/whisper",purpose:"Speech-to-text",note:"Chạy được CPU; API đơn giản",cat:"Audio"},{name:"facebookresearch/fairseq (HuBERT)",url:"https://github.com/facebookresearch/fairseq/tree/main/examples/hubert",purpose:"Audio encoder self-supervised",note:"Nặng, chỉ cần hiểu cách dùng",cat:"Audio"},{name:"huggingface/diffusers",url:"https://github.com/huggingface/diffusers",purpose:"Diffusion pipelines",note:"Thử nghiệm sinh ảnh",cat:"CV"},{name:"karpathy/nanoGPT",url:"https://github.com/karpathy/nanoGPT",purpose:"GPT viết lại từ đầu, code ngắn",note:"Học transformer thực chiến",cat:"Học tập"},{name:"mrdbourke/pytorch-deep-learning",url:"https://github.com/mrdbourke/pytorch-deep-learning",purpose:"Notebook PyTorch bài bản",note:"Học theo tuần 20+",cat:"Học tập"},{name:"d2l-ai/d2l-en",url:"https://github.com/d2l-ai/d2l-en",purpose:"Sách Deep Learning kèm code",note:"Tra cứu mỗi khi quên lý thuyết",cat:"Học tập"},{name:"microsoft/ML-For-Beginners",url:"https://github.com/microsoft/ML-For-Beginners",purpose:"Giáo trình 12 tuần ML",note:"Ôn ML cổ điển",cat:"Học tập"},{name:"scikit-learn/scikit-learn",url:"https://github.com/scikit-learn/scikit-learn",purpose:"ML cổ điển",note:"Đọc API + ví dụ",cat:"Tiện ích"},{name:"optuna/optuna",url:"https://github.com/optuna/optuna",purpose:"Tuning hyperparameter",note:"Dùng trong notebook thi",cat:"Tiện ích"},{name:"albumentations-team/albumentations",url:"https://github.com/albumentations-team/albumentations",purpose:"Augmentation ảnh",note:"Nhanh, nhiều phép biến đổi",cat:"Tiện ích"},{name:"IOAI-official/IOAI-2026",url:"https://github.com/IOAI-official/IOAI-2026",purpose:"Đề + baseline chính thức IOAI 2026",note:"Tài liệu luyện tập số 1",cat:"Đề thi"},{name:"IOAI-official/IOAI-2025",url:"https://github.com/IOAI-official/IOAI-2025",purpose:"Đề IOAI 2025",note:"Làm trong đúng 6h",cat:"Đề thi"},{name:"IOAI-official/IOAI-2024",url:"https://github.com/IOAI-official/IOAI-2024",purpose:"Đề IOAI 2024",note:"Bắt đầu từ các task dễ",cat:"Đề thi"}],l=[{id:"s1",title:"Khung training loop PyTorch chuẩn",when:"Mọi bài deep learning (tuần 20+)",code:`def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total`},{id:"s2",title:"Dataset ảnh từ thư mục (ImageFolder)",when:"Bài CV dạng phân loại ảnh (tuần 26+)",code:`from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_ds = datasets.ImageFolder('data/train', transform=tf)
val_ds = datasets.ImageFolder('data/val', transform=tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)`},{id:"s3",title:"Transfer learning với timm",when:"Bài CV khi cần điểm nhanh (tuần 27+)",code:`import timm, torch.nn as nn

model = timm.create_model('resnet18', pretrained=True, num_classes=NUM_CLASSES)
# hoặc ViT: timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=N)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)`},{id:"s4",title:"Fine-tune text classification với HuggingFace",when:"Bài NLP (tuần 33+)",code:`from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tok = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-multilingual-cased', num_labels=2)

enc = tok(list(texts), truncation=True, padding=True, max_length=128, return_tensors='pt')

args = TrainingArguments(output_dir='out', num_train_epochs=3,
                         per_device_train_batch_size=16, learning_rate=2e-5,
                         evaluation_strategy='epoch', save_strategy='no')
trainer = Trainer(model=model, args=args, train_dataset=ds, eval_dataset=val_ds)
trainer.train()`},{id:"s5",title:"Mel-spectrogram với torchaudio",when:"Bài audio (tuần 35+)",code:`import torch, torchaudio

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)
to_db = torchaudio.transforms.AmplitudeToDB()

waveform, sr = torchaudio.load('clip.wav')
feat = to_db(mel(waveform))  # [1, 64, T] -> đưa vào CNN`},{id:"s6",title:"Kiểm tra format file nộp",when:"Trước mọi lần nộp (tuần 43+)",code:`import pandas as pd

sub = pd.read_csv('submission.csv')
assert list(sub.columns) == ['id', 'label'], sub.columns
assert sub['id'].is_unique
assert sub['label'].notna().all()
assert len(sub) == EXPECTED_ROWS, len(sub)
print('OK', sub.head())`},{id:"s7",title:"Chia dữ liệu & cross-validation",when:"ML cổ điển (tuần 11+)",code:`from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_tr, y_tr, cv=skf, scoring='f1_macro')
print(scores.mean(), scores.std())`},{id:"s8",title:"Seed toàn bộ để tái lập kết quả",when:"Đầu mọi notebook thi",code:`import random, numpy as np, torch, os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)`},{id:"s9",title:"Tiết kiệm RAM trên máy 4GB",when:"Khi EDA/ train nhẹ trên máy yếu (tuần 4+)",code:`import gc
import pandas as pd

# 1) Đọc từng chunk thay vì cả file lớn
for chunk in pd.read_csv('big.csv', chunksize=100_000, usecols=['a', 'b', 'label']):
    process(chunk)  # xử lý rồi bỏ, không giữ lại

# 2) Giảm bộ nhớ cột chuỗi lặp lại + số nhỏ
df['city'] = df['city'].astype('category')
df['label'] = df['label'].astype('int8')

# 3) Xoá biến lớn khi không cần nữa
del big_df
gc.collect()

# 4) Đo bộ nhớ đang dùng (MB)
print(df.memory_usage(deep=True).sum() / 1e6, 'MB')`},{id:"s10",title:"Kiểm tra môi trường máy yếu / có GPU hay không",when:"Đầu tuần 20 khi chuyển sang PyTorch",code:`import sys
print(sys.version)

import numpy, pandas, sklearn
print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| sklearn', sklearn.__version__)

# Máy 4GB thường KHÔNG có CUDA — điều này bình thường:
try:
    import torch
    print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())
except ImportError:
    print('Chưa cài torch — máy yếu nên dùng Colab/Kaggle cho deep learning')`}];function _(){const[s,i]=a.useState("repo"),[n,c]=a.useState("all"),[m,r]=a.useState(null),d=a.useMemo(()=>[...new Set(o.map(t=>t.cat))],[]),h=a.useMemo(()=>n==="all"?o:o.filter(t=>t.cat===n),[n]);async function u(t,p){try{await navigator.clipboard.writeText(p),r(t),window.setTimeout(()=>r(null),1500)}catch{r(null)}}return e.jsxs(e.Fragment,{children:[e.jsxs("div",{className:"card",children:[e.jsx("h2",{style:{marginTop:0},children:"🧰 Toolkit luyện thi"}),e.jsx("p",{className:"muted small",style:{marginTop:0},children:"Kho GitHub open-source và các đoạn code tái sử dụng. Trong phòng thi không có internet — hãy học cách dùng và chép vào “sổ tay code” cá nhân trước."}),e.jsxs("div",{className:"row",children:[e.jsxs("button",{className:`btn ${s==="repo"?"primary":""}`,onClick:()=>i("repo"),children:["GitHub (",o.length,")"]}),e.jsxs("button",{className:`btn ${s==="code"?"primary":""}`,onClick:()=>i("code"),children:["Đoạn code (",l.length,")"]})]})]}),s==="repo"?e.jsxs(e.Fragment,{children:[e.jsx("div",{className:"card",children:e.jsxs("div",{className:"row",children:[e.jsx("button",{className:`btn sm ${n==="all"?"primary":""}`,onClick:()=>c("all"),children:"Tất cả"}),d.map(t=>e.jsx("button",{className:`btn sm ${n===t?"primary":""}`,onClick:()=>c(t),children:t},t))]})}),h.map(t=>e.jsxs("div",{className:"card",children:[e.jsxs("div",{className:"row between",children:[e.jsxs("div",{children:[e.jsx("b",{children:t.name}),e.jsxs("div",{className:"tiny muted",children:[e.jsx("span",{className:"badge blue",children:t.cat})," ",t.purpose]})]}),e.jsx("a",{className:"btn sm primary",href:t.url,target:"_blank",rel:"noreferrer",children:"Mở GitHub"})]}),e.jsxs("div",{className:"small muted",style:{marginTop:6},children:["💡 ",t.note]})]},t.name))]}):e.jsx(e.Fragment,{children:l.map(t=>e.jsxs("div",{className:"card",children:[e.jsxs("div",{className:"row between",children:[e.jsxs("div",{children:[e.jsx("b",{children:t.title}),e.jsxs("div",{className:"tiny muted",children:["Dùng khi: ",t.when]})]}),e.jsx("button",{className:"btn sm",onClick:()=>void u(t.id,t.code),children:m===t.id?"✅ Đã chép":"📋 Chép code"})]}),e.jsx("pre",{className:"code",children:t.code})]},t.id))})]})}export{_ as default};
