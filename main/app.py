#FCN8S
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import threading

# Web server imports
import os
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template, abort
from werkzeug.utils import secure_filename

########################## FCN8s & UNet Architectures ##########################


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        return self.out(dec1)
class FCN8s(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(FCN8s, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        # FC layers as convolutions
        fc6 = self.relu6(self.fc6(pool5))
        fc6 = self.drop6(fc6)

        fc7 = self.relu7(self.fc7(fc6))
        fc7 = self.drop7(fc7)

        # Score
        score_fr = self.score_fr(fc7)

        # Upsample and fuse
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        upscore2 = self._crop(upscore2, score_pool4)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(pool3)
        upscore_pool4 = self._crop(upscore_pool4, score_pool3)
        fuse_pool3 = upscore_pool4 + score_pool3

        # Final upsampling
        out = self.upscore8(fuse_pool3)
        out = self._crop(out, x)

        return out

    def _crop(self, input_tensor, target_tensor):
        _, _, h, w = target_tensor.size()
        return input_tensor[:, :, :h, :w]

# ==================== GPU REPORTING HELPER ====================
def report_gpu(prefix=''):
    """Print GPU device info and memory usage."""
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        print(f"{prefix} CUDA device: {i} - {torch.cuda.get_device_name(i)}")
        print(f"{prefix} capability: {torch.cuda.get_device_capability(i)}")
        print(f"{prefix} allocated: {torch.cuda.memory_allocated(i)/1024**2:.1f} MB")
        print(f"{prefix} reserved:  {torch.cuda.memory_reserved(i)/1024**2:.1f} MB")
    else:
        print(f"{prefix} CUDA not available; running on CPU.")

# ==================== AUTO-DETECT NUM_CLASSES ====================
def detect_num_classes(model_path, model_type='fcn8s'):
    """Auto-detect number of classes from model file for FCN8s or UNet."""
    state_dict = torch.load(model_path, map_location='cpu')
    if model_type == 'fcn8s':
        for key in state_dict.keys():
            if "score_fr.weight" in key:
                return state_dict[key].shape[0]
        raise ValueError("❌ Tidak menemukan score_fr.weight → bukan FCN-8s?")
    elif model_type == 'unet':
        # Try common keys for UNet
        for key in state_dict.keys():
            if "out.weight" in key:
                return state_dict[key].shape[0]
        # Fallback: largest first dim
        tensors = [v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.ndim >= 1]
        if tensors:
            return max(int(t.shape[0]) for t in tensors)
        raise ValueError("❌ Tidak menemukan out.weight → bukan UNet?")
    else:
        raise ValueError("Unknown model_type for num_classes detection")

# ==================== VIDEO PREDICTION FUNCTION ====================
def predict_video_fcn8s(model_path, video_path, output_path=None, num_classes=None, 
                        resize_dim=(256, 256), skip_frames=1, show_overlay=True, 
                        batch_size=4, half_precision=True):
    """
    Prediksi segmentasi video menggunakan FCN-8s dengan batch processing
    
    Args:
        model_path: Path ke model FCN-8s (.pth)
        video_path: Path ke video input
        output_path: Path untuk menyimpan video output (default: auto-generated)
        num_classes: Jumlah kelas (None = auto-detect)
        resize_dim: Dimensi untuk resize frame (default: 256x256)
        skip_frames: Proses setiap N frame (1 = semua frame, 2 = setiap 2 frame, dst)
        show_overlay: Tampilkan overlay atau hanya mask
        batch_size: Jumlah frame diproses sekaligus
        half_precision: Gunakan FP16 untuk mempercepat inference
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto detect jumlah kelas
    if num_classes is None:
        print("🔍 Auto-detecting number of classes...")
        num_classes = detect_num_classes(model_path)
        print(f"✅ Detected: {num_classes} classes\n")
    
    print("="*60)
    print("🎬 VIDEO SEGMENTATION MODE (FCN-8s)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Classes: {num_classes}")
    print(f"Resize: {resize_dim}")
    print(f"Skip frames: {skip_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Half precision (FP16): {half_precision}")
    print("="*60, "\n")
    
    # Load model
    print("📦 Loading model...")
    model = FCN8s(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Convert to half precision if enabled
    if half_precision and device.type == 'cuda':
        model = model.half()
        print("✅ Model loaded with FP16 optimization!")
        report_gpu("after model.half():")
    else:
        print("✅ Model loaded!")
        report_gpu("after model.to(device):")
    print()
    
    # Open video
    print("🎥 Opening video...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("❌ Tidak bisa membuka video!")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Info:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Duration: {total_frames/fps:.2f}s\n")
    
    # Setup output video
    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + '_segmented.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # ✅ PENTING: Tulis dengan FPS asli agar durasi sama dengan video input
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"💾 Output will be saved to: {output_path}\n")
    print("🚀 Processing video...\n")
    
    # Process video
    frame_count = 0
    processed_count = 0
    skipped_count = 0
    frames_buffer = []
    predictions_buffer = []
    
    with tqdm(total=total_frames, desc="Processing", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # ONLY process and write every skip_frames (skip lainnya)
            if frame_count % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append((frame, frame_rgb))
                
                # Process batch when buffer is full
                if len(frames_buffer) >= batch_size:
                    # Prepare batch
                    batch_frames = []
                    for _, fr in frames_buffer:
                        fr_resized = cv2.resize(fr, resize_dim)
                        fr_np = fr_resized / 255.0
                        batch_frames.append(fr_np)
                    
                    batch_tensor = torch.FloatTensor(np.array(batch_frames)).permute(0, 3, 1, 2).to(device)
                    
                    if half_precision and device.type == 'cuda':
                        batch_tensor = batch_tensor.half()
                    
                    # Batch inference
                    with torch.no_grad():
                        outputs = model(batch_tensor)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    # Store predictions
                    for pred in preds:
                        pred_resized = cv2.resize(pred.astype(np.uint8), (width, height), 
                                                 interpolation=cv2.INTER_NEAREST)
                        predictions_buffer.append(pred_resized)
                    
                    # Write frames (ONLY processed frames)
                    for i, (orig_frame, orig_rgb) in enumerate(frames_buffer):
                        pred_resized = predictions_buffer[i]
                        
                        if show_overlay:
                            mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                            mask_color = (mask_color * 255).astype(np.uint8)
                            result = cv2.addWeighted(orig_rgb, 0.6, mask_color, 0.4, 0)
                        else:
                            mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                            result = (mask_color * 255).astype(np.uint8)
                        
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        out.write(result_bgr)
                    
                    processed_count += len(frames_buffer)
                    frames_buffer = []
                    predictions_buffer = []
            else:
                # Skip frame - don't write to output
                skipped_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        # Process remaining frames in buffer
        if frames_buffer:
            batch_frames = []
            for _, fr in frames_buffer:
                fr_resized = cv2.resize(fr, resize_dim)
                fr_np = fr_resized / 255.0
                batch_frames.append(fr_np)
            
            batch_tensor = torch.FloatTensor(np.array(batch_frames)).permute(0, 3, 1, 2).to(device)
            
            if half_precision and device.type == 'cuda':
                batch_tensor = batch_tensor.half()
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for i, (orig_frame, orig_rgb) in enumerate(frames_buffer):
                pred_resized = cv2.resize(preds[i].astype(np.uint8), (width, height), 
                                         interpolation=cv2.INTER_NEAREST)
                
                if show_overlay:
                    mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                    mask_color = (mask_color * 255).astype(np.uint8)
                    result = cv2.addWeighted(orig_rgb, 0.6, mask_color, 0.4, 0)
                else:
                    mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                    result = (mask_color * 255).astype(np.uint8)
                
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                out.write(result_bgr)
            
            processed_count += len(frames_buffer)
    
    # Cleanup
    cap.release()
    out.release()
    
    print("\n" + "="*60)
    print("✨ PROCESSING COMPLETE!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   - Total frames read: {total_frames}")
    print(f"   - Frames processed: {processed_count}")
    print(f"   - Frames skipped: {skipped_count}")
    print(f"   - Skip ratio: 1:{skip_frames}")
    print(f"   - Output frames: {processed_count}")
    print(f"   - Output FPS: {fps}")
    print(f"   - Output duration: {processed_count/fps:.2f}s")
    print(f"   - Output: {output_path}")
    print("="*60)
    
    return output_path

# ==================== SIMPLE IMAGE PREDICTION HELPERS + FLASK API ====================



ALLOWED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov'}

def is_image_filename(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_IMAGE_EXTS

def is_video_filename(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO_EXTS



def predict_image(model, device, num_classes, image_path, output_path,
                 resize_dim=(256, 256), show_overlay=True):
    """Run segmentation model on a single image and save overlay result to output_path."""
    pil = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil.size
    img_np = np.array(pil)
    img_resized = cv2.resize(img_np, resize_dim)
    img_in = img_resized.astype(np.float32) / 255.0
    tensor = torch.FloatTensor(img_in).permute(2, 0, 1).unsqueeze(0).to(device)
    if next(model.parameters()).dtype == torch.float16:
        tensor = tensor.half()
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0].astype(np.uint8)
    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    cmap = plt.cm.tab20
    mask_color = (cmap(pred_resized / max(num_classes-1, 1))[:, :, :3] * 255).astype(np.uint8)
    if show_overlay:
        overlay = cv2.addWeighted(img_np, 0.6, mask_color, 0.4, 0)
    else:
        overlay = mask_color
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path



# Flask app and model management
app = Flask(__name__, template_folder='templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model state
MODEL = None
MODEL_NUM_CLASSES = None
MODEL_LOADED = False
MODEL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Video job management
# Video job management
VIDEO_JOBS = {}  # job_id: {status, progress, output_path, error}
VIDEO_JOBS_LOCK = threading.Lock()

# Model config
MODEL_CONFIG = {
    'fcn8s': {
        'model_path': 'C:\\Users\\agust\\OneDrive\\Desktop\\project3 final\\model\\best_fcn8s_model.pth',
        'class': FCN8s
    },
    'unet': {
        'model_path': 'C:\\Users\\agust\\OneDrive\\Desktop\\project3 final\\model\\best_unet_model.pth',
        'class': UNet
    }
}



def load_model(model_type='fcn8s'):
    config = MODEL_CONFIG.get(model_type)
    if not config:
        raise ValueError('Unknown model_type')
    model_path = os.path.join(BASE_DIR, config['model_path'])
    if not os.path.exists(model_path):
        return None, None
    num_classes = detect_num_classes(model_path, model_type)
    model = config['class'](num_classes=num_classes).to(MODEL_DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=MODEL_DEVICE))
    model.eval()
    if MODEL_DEVICE.type == 'cuda':
        model = model.half()
    return model, num_classes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = 'cuda'

    return jsonify({
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'model_loaded': MODEL_LOADED,
        'num_classes': MODEL_NUM_CLASSES
    })



@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    file.save(upload_path)

    # Read extra params
    resize_dim = int(request.form.get('resize_dim', 256))
    skip_frames = int(request.form.get('skip_frames', 1))
    batch_size = int(request.form.get('batch_size', 8))
    model_type = request.form.get('model_type', 'fcn8s').lower()

    # Image file
    if is_image_filename(filename):
        model, num_classes = load_model(model_type)
        if model is None:
            return jsonify({'error': f'Model weights not found for {model_type}.'}), 500
        out_name = f"{uuid.uuid4().hex}_seg.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        try:
            predict_image(model, MODEL_DEVICE, num_classes,
                          upload_path, out_path, resize_dim=(resize_dim, resize_dim),
                          show_overlay=True)
            return jsonify({'output_path': f'/outputs/{out_name}'}), 200
        except Exception as e:
            print('Processing error:', e)
            return jsonify({'error': str(e)}), 500

    # Video file
    elif is_video_filename(filename):
        model, num_classes = load_model(model_type)
        if model is None:
            return jsonify({'error': f'Model weights not found for {model_type}.'}), 500
        job_id = uuid.uuid4().hex
        out_name = f"{job_id}_seg.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with threading.Lock():
            VIDEO_JOBS[job_id] = {
                'status': 'queued',
                'progress': 0,
                'output_path': None,
                'error': None
            }
        def video_job():
            try:
                with threading.Lock():
                    VIDEO_JOBS[job_id]['status'] = 'processing'
                    VIDEO_JOBS[job_id]['progress'] = 0
                class TqdmJob:
                    def __init__(self, total, *args, **kwargs):
                        self.total = total
                        self.n = 0
                    def update(self, n):
                        self.n += n
                        with threading.Lock():
                            VIDEO_JOBS[job_id]['progress'] = min(100, int(100 * self.n / self.total))
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc_val, exc_tb): pass
                orig_tqdm = globals().get('tqdm', None)
                globals()['tqdm'] = TqdmJob
                try:
                    if model_type == 'fcn8s':
                        result_path = predict_video_fcn8s(
                            model_path=MODEL_CONFIG['fcn8s']['model_path'],
                            video_path=upload_path,
                            output_path=out_path,
                            num_classes=num_classes,
                            resize_dim=(resize_dim, resize_dim),
                            skip_frames=skip_frames,
                            show_overlay=True,
                            batch_size=batch_size,
                            half_precision=(MODEL_DEVICE.type == 'cuda')
                        )
                    elif model_type == 'unet':
                        result_path = predict_video_unet(
                            model_path=MODEL_CONFIG['unet']['model_path'],
                            video_path=upload_path,
                            output_path=out_path,
                            num_classes=num_classes,
                            resize_dim=(resize_dim, resize_dim),
                            skip_frames=skip_frames,
                            show_overlay=True,
                            batch_size=batch_size,
                            half_precision=(MODEL_DEVICE.type == 'cuda')
                        )
                    else:
                        raise ValueError('Unknown model_type')
                finally:
                    if orig_tqdm:
                        globals()['tqdm'] = orig_tqdm
                with threading.Lock():
                    VIDEO_JOBS[job_id]['status'] = 'done'
                    VIDEO_JOBS[job_id]['progress'] = 100
                    VIDEO_JOBS[job_id]['output_path'] = f'/outputs/{out_name}'
            except Exception as e:
                with threading.Lock():
                    VIDEO_JOBS[job_id]['status'] = 'error'
                    VIDEO_JOBS[job_id]['error'] = str(e)
        threading.Thread(target=video_job, daemon=True).start()
        return jsonify({'job_id': job_id}), 202
    else:
        return jsonify({'error': 'File type not supported. Only images and videos.'}), 400
# ...existing code...

def predict_video_unet(model_path, video_path, output_path=None, num_classes=None,
                      resize_dim=(256, 256), skip_frames=1, show_overlay=True,
                      batch_size=4, half_precision=True):
    """Segment video using U-Net (batch, overlay, skip frames, FP16)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if num_classes is None:
        num_classes = detect_num_classes(model_path, 'unet')
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if half_precision and device.type == 'cuda':
        model = model.half()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Tidak bisa membuka video!")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + '_segmented.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    processed_count = 0
    skipped_count = 0
    frames_buffer = []
    predictions_buffer = []
    with tqdm(total=total_frames, desc="Processing", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append((frame, frame_rgb))
                if len(frames_buffer) >= batch_size:
                    batch_frames = []
                    for _, fr in frames_buffer:
                        fr_resized = cv2.resize(fr, resize_dim)
                        fr_np = fr_resized / 255.0
                        batch_frames.append(fr_np)
                    batch_tensor = torch.FloatTensor(np.array(batch_frames)).permute(0, 3, 1, 2).to(device)
                    if half_precision and device.type == 'cuda':
                        batch_tensor = batch_tensor.half()
                    with torch.no_grad():
                        outputs = model(batch_tensor)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    for pred in preds:
                        pred_resized = cv2.resize(pred.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                        predictions_buffer.append(pred_resized)
                    for i, (orig_frame, orig_rgb) in enumerate(frames_buffer):
                        pred_resized = predictions_buffer[i]
                        if show_overlay:
                            mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                            mask_color = (mask_color * 255).astype(np.uint8)
                            result = cv2.addWeighted(orig_rgb, 0.6, mask_color, 0.4, 0)
                        else:
                            mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                            result = (mask_color * 255).astype(np.uint8)
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        out.write(result_bgr)
                    processed_count += len(frames_buffer)
                    frames_buffer = []
                    predictions_buffer = []
            else:
                skipped_count += 1
            frame_count += 1
            pbar.update(1)
        if frames_buffer:
            batch_frames = []
            for _, fr in frames_buffer:
                fr_resized = cv2.resize(fr, resize_dim)
                fr_np = fr_resized / 255.0
                batch_frames.append(fr_np)
            batch_tensor = torch.FloatTensor(np.array(batch_frames)).permute(0, 3, 1, 2).to(device)
            if half_precision and device.type == 'cuda':
                batch_tensor = batch_tensor.half()
            with torch.no_grad():
                outputs = model(batch_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            for i, (orig_frame, orig_rgb) in enumerate(frames_buffer):
                pred_resized = cv2.resize(preds[i].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                if show_overlay:
                    mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                    mask_color = (mask_color * 255).astype(np.uint8)
                    result = cv2.addWeighted(orig_rgb, 0.6, mask_color, 0.4, 0)
                else:
                    mask_color = plt.cm.tab20(pred_resized / max(num_classes-1, 1))[:, :, :3]
                    result = (mask_color * 255).astype(np.uint8)
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                out.write(result_bgr)
            processed_count += len(frames_buffer)
    cap.release()
    out.release()
    return output_path
# ...existing code...

@app.route('/api/job_status/<job_id>')
def api_job_status(job_id):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    safe_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(safe_path):
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename)


# ==================== ENTRYPOINT: RUN FLASK APP ====================
if __name__ == "__main__":
    # Preload model at startup
    print("Starting server...")
    # Log device info
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("⚠️ CUDA not available; running on CPU.")

    try:
        MODEL, MODEL_NUM_CLASSES = load_model('fcn8s')
        MODEL_LOADED = MODEL is not None
        print(f"Model loaded: {MODEL_LOADED}, num_classes: {MODEL_NUM_CLASSES}")
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL = None
        MODEL_NUM_CLASSES = None
        MODEL_LOADED = False
    app.run(host='0.0.0.0', port=5000, debug=True)