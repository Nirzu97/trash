import streamlit as st
import numpy as np
import cv2
import os
import random
from PIL import Image
from ultralytics import YOLO
# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SwachhBharat AI | Intelligent Trash Segregator",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Apply premium modern styling (custom CSS for dark mode and glassmorphic UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main block styling */
    .stApp {
        background-color: #0f1115;
        color: #e2e8f0;
    }
    
    /* Custom headers and subheaders */
    h1 {
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        margin-bottom: 5px !important;
        letter-spacing: -0.05rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
    }
    
    /* Metric Card Styling */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-top: 4px;
    }
    
    /* SWM Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-bottom: 8px;
    }
    
    .badge-green { background-color: #2E7D32; }
    .badge-blue { background-color: #1565C0; }
    .badge-red { background-color: #C62828; }
    .badge-black { background-color: #212121; border: 1px solid #444; }
    .badge-gray { background-color: #757575; }
    
    /* Sidebar premium edits */
    section[data-testid="stSidebar"] {
        background-color: #131720;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# Class Mappings (SWM Rules, 2026 Guidelines)
# -----------------------------------------------------------------------------
CLASS_MAPPING = {
    'Brick': {'category': 'Construction & Demolition Waste', 'bin': 'Gray Bin', 'color_rgb': (117, 117, 117), 'hex': '#757575', 'instruction': 'Store separately. Hand over to municipal C&D waste collector or local recycling facility.'},
    'Broken Glass': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Rinse off residues. Wrap broken glass securely in paper to prevent injury to waste handlers before disposing.'},
    'Cardboard': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Flatten boxes to save space. Keep dry.'},
    'Cigarette': {'category': 'Domestic Hazardous Waste', 'bin': 'Black Bin', 'color_rgb': (33, 33, 33), 'hex': '#212121', 'instruction': 'Ensure completely extinguished. Dispose in black hazardous waste bin.'},
    'Cloth': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Donate if usable. If torn/unusable, place in dry waste bin for textile recycling.'},
    'Coconut Shell': {'category': 'Wet Waste (Organic)', 'bin': 'Green Bin', 'color_rgb': (46, 125, 50), 'hex': '#2E7D32', 'instruction': 'Biodegradable. Suitable for home or community composting.'},
    'Dairy Packets': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Cut open and rinse thoroughly to remove milk residue (prevents odor/pests). Dry before throwing.'},
    'Footwear': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Donate if in decent condition. Otherwise, dispose in dry waste.'},
    'Glass Bottle': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Rinse out contents. Glass is 100% infinitely recyclable.'},
    'Mask': {'category': 'Sanitary Waste', 'bin': 'Red Bin', 'color_rgb': (198, 40, 40), 'hex': '#C62828', 'instruction': 'Cut the ear loops (prevents animal entanglement). Wrap securely in a paper bag marked with a red mark/dot before disposal.'},
    'Medical Waste': {'category': 'Domestic Hazardous Waste', 'bin': 'Black Bin', 'color_rgb': (33, 33, 33), 'hex': '#212121', 'instruction': 'Includes expired medicines, syringes, etc. Dispose safely in black hazardous waste bin or drop off at local pharmacy collections.'},
    'Metal Can': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Rinse out beverage/food residue. Crush to save space.'},
    'Packaging Box': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Flatten boxes. Remove plastic tapes/bubble wrap and segregate them separately.'},
    'Paper': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Ensure paper is dry and not soiled by food. Shreddable.'},
    'Paper Bag': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Reuse as much as possible. If torn, discard in blue bin.'},
    'Paper Cup': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Note: Often coated with a thin plastic film. Check if clean, otherwise dump in dry waste.'},
    'Paper Utensils': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Dispose only if dry and free from grease/food residue.'},
    'Plastic Bag': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Reuse bags. Ensure dry and clean. Thin single-use plastics are banned in many Indian states.'},
    'Plastic Bits': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Small plastic scraps. Collect inside a larger plastic bottle/bag before disposal to prevent environmental dispersion.'},
    'Plastic Bottle': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Empty contents, crush the bottle, and screw the cap back on (or put the cap inside the bottle).'},
    'Plastic Box': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Clean container before disposal. Highly recyclable.'},
    'Plastic Cup': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Rinse out liquids. Check local plastic grading.'},
    'Plastic Packet': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Multi-layer packaging (chips packets, etc.). Send to dry waste; often used for waste-to-energy or road construction.'},
    'Plastic Straw': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Avoid using single-use straws. Discard in dry waste if used.'},
    'Plastic Utensils': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Wash off food remnants before disposing in dry waste.'},
    'Sanitary': {'category': 'Sanitary Waste', 'bin': 'Red Bin', 'color_rgb': (198, 40, 40), 'hex': '#C62828', 'instruction': 'Includes diapers, sanitary pads. Wrap securely in a paper pouch/newspaper marked with a red mark before throwing into sanitary waste.'},
    'Synthetic Bag': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Made of synthetic polymers (polyester/nylon). Recyclable, place in blue bin.'},
    'Thermocol': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Expanded polystyrene. Place in dry waste. Handle carefully as it breaks easily and pollutes ecosystems.'},
    'Tile': {'category': 'Construction & Demolition Waste', 'bin': 'Gray Bin', 'color_rgb': (117, 117, 117), 'hex': '#757575', 'instruction': 'Do not mix with municipal organic or dry waste. Hand over to construction waste collectors.'},
    'Tobacco Packet': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Usually multi-layer plastic packaging. Place in blue bin.'},
    'Wood Materials': {'category': 'Dry Waste (Recyclable)', 'bin': 'Blue Bin', 'color_rgb': (21, 101, 192), 'hex': '#1565C0', 'instruction': 'Dry wood waste. Can be recycled or repurposed. Place in dry waste.'}
}
CLASSES = list(CLASS_MAPPING.keys())
# -----------------------------------------------------------------------------
# Model Loader & Mock Detector
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        try:
            return YOLO(model_path), True
        except Exception as e:
            return None, False
    return None, False
model, model_loaded = load_model()
def get_bin_badge(bin_name):
    if bin_name == "Green Bin":
        return f'<span class="badge badge-green">{bin_name}</span>'
    elif bin_name == "Blue Bin":
        return f'<span class="badge badge-blue">{bin_name}</span>'
    elif bin_name == "Red Bin":
        return f'<span class="badge badge-red">{bin_name}</span>'
    elif bin_name == "Black Bin":
        return f'<span class="badge badge-black">{bin_name}</span>'
    else:
        return f'<span class="badge badge-gray">{bin_name}</span>'
# Mock predictions for simulation mode
def get_mock_predictions(image_name, img_w, img_h):
    detections = []
    if "plastic_bottle" in image_name:
        poly = [
            [int(0.12 * img_w), int(0.52 * img_h)],
            [int(0.20 * img_w), int(0.32 * img_h)],
            [int(0.38 * img_w), int(0.28 * img_h)],
            [int(0.72 * img_w), int(0.38 * img_h)],
            [int(0.85 * img_w), int(0.50 * img_h)],
            [int(0.92 * img_w), int(0.62 * img_h)],
            [int(0.80 * img_w), int(0.72 * img_h)],
            [int(0.55 * img_w), int(0.68 * img_h)],
            [int(0.25 * img_w), int(0.62 * img_h)]
        ]
        detections.append({
            'class': 'Plastic Bottle',
            'confidence': 0.96,
            'polygon': poly
        })
    elif "cardboard_box" in image_name:
        poly = [
            [int(0.22 * img_w), int(0.32 * img_h)],
            [int(0.48 * img_w), int(0.12 * img_h)],
            [int(0.82 * img_w), int(0.16 * img_h)],
            [int(0.92 * img_w), int(0.32 * img_h)],
            [int(0.78 * img_w), int(0.85 * img_h)],
            [int(0.25 * img_w), int(0.88 * img_h)],
            [int(0.18 * img_w), int(0.55 * img_h)]
        ]
        detections.append({
            'class': 'Cardboard',
            'confidence': 0.92,
            'polygon': poly
        })
    elif "metal_can" in image_name:
        poly = [
            [int(0.20 * img_w), int(0.52 * img_h)],
            [int(0.28 * img_w), int(0.28 * img_h)],
            [int(0.48 * img_w), int(0.25 * img_h)],
            [int(0.75 * img_w), int(0.30 * img_h)],
            [int(0.80 * img_w), int(0.68 * img_h)],
            [int(0.68 * img_w), int(0.72 * img_h)],
            [int(0.32 * img_w), int(0.70 * img_h)]
        ]
        detections.append({
            'class': 'Metal Can',
            'confidence': 0.94,
            'polygon': poly
        })
    elif "face_mask" in image_name:
        poly = [
            [int(0.20 * img_w), int(0.58 * img_h)],
            [int(0.28 * img_w), int(0.38 * img_h)],
            [int(0.48 * img_w), int(0.30 * img_h)],
            [int(0.70 * img_w), int(0.35 * img_h)],
            [int(0.84 * img_w), int(0.56 * img_h)],
            [int(0.78 * img_w), int(0.72 * img_h)],
            [int(0.55 * img_w), int(0.68 * img_h)],
            [int(0.22 * img_w), int(0.72 * img_h)]
        ]
        detections.append({
            'class': 'Mask',
            'confidence': 0.97,
            'polygon': poly
        })
    elif "coconut_shell" in image_name:
        poly = [
            [int(0.12 * img_w), int(0.50 * img_h)],
            [int(0.22 * img_w), int(0.25 * img_h)],
            [int(0.45 * img_w), int(0.22 * img_h)],
            [int(0.55 * img_w), int(0.48 * img_h)],
            [int(0.52 * img_w), int(0.72 * img_h)],
            [int(0.25 * img_w), int(0.74 * img_h)]
        ]
        detections.append({
            'class': 'Coconut Shell',
            'confidence': 0.95,
            'polygon': poly
        })
    else:
        # Default mock detection for custom uploaded images in simulation mode
        # Generate 1-2 random detections inside the image center
        classes_pool = ['Plastic Bottle', 'Paper', 'Metal Can', 'Mask', 'Coconut Shell', 'Glass Bottle']
        num_dets = random.randint(1, 2)
        for i in range(num_dets):
            c_name = random.choice(classes_pool)
            cx, cy = img_w // 2, img_h // 2
            rw, rh = img_w // 4, img_h // 4
            if i > 0:
                cx += img_w // 6
                cy += img_h // 6
            poly = [
                [cx - rw, cy - rh],
                [cx + rw, cy - rh],
                [cx + rw, cy + rh],
                [cx - rw, cy + rh]
            ]
            detections.append({
                'class': c_name,
                'confidence': round(random.uniform(0.82, 0.95), 2),
                'polygon': poly
            })
    return detections
# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.markdown("<h1>♻️ SwachhBharat AI</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Trash Segregator powered by YOLOv8 Segmentation & SWM Rules 2026</p>', unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.markdown('<div class="glass-card" style="padding: 16px;">', unsafe_allow_html=True)
    st.markdown("### ⚙️ System Status")
    if model_loaded:
        st.markdown('<p style="color: #10b981; font-weight: 600; margin-bottom: 2px;">🟢 YOLOv8 Engine Active</p>', unsafe_allow_html=True)
        st.caption("Running live inference with `best.pt`")
    else:
        st.markdown('<p style="color: #f59e0b; font-weight: 600; margin-bottom: 2px;">🟡 Simulation Mode Active</p>', unsafe_allow_html=True)
        st.caption("No `best.pt` found. Running interactive SWM simulation.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="padding: 16px;">', unsafe_allow_html=True)
    st.markdown("### 🛢️ Segregation Guide (SWM 2026)")
    st.markdown("""
    - 🟢 **Green Bin**: Wet / Organic / Kitchen Waste
    - 🔵 **Blue Bin**: Dry / Recyclables (Paper, Plastic, Metal, Glass)
    - 🔴 **Red Bin**: Sanitary Waste (Masks, Hygiene Products)
    - ⚫ **Black Bin**: Domestic Hazardous (Medicines, Cigarettes, e-waste)
    - ⚪ **Gray Bin**: Construction & Demolition (C&D) Waste (Bricks, Tiles)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("Developed in compliance with the Swachh Bharat Mission (Urban & Rural) Solid Waste Management Rules, 2026.")
# Tabs for Mode of Input
tab1, tab2 = st.tabs(["📁 Try Live Detector", "🎨 Example Sandbox"])
# Processing Function
def process_and_display(img_pil, image_name="custom_upload"):
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = img_bgr.shape
    
    # Run Inference
    detected_items = []
    
    if model_loaded:
        with st.spinner("Analyzing image details..."):
            results = model.predict(img_bgr, conf=0.25)
            
        result = results[0]
        if result.masks is not None:
            for mask_polygon, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                conf = float(box.conf[0].item())
                
                # Check if class matches mapping
                if class_name in CLASS_MAPPING:
                    detected_items.append({
                        'class': class_name,
                        'confidence': conf,
                        'polygon': np.array(mask_polygon, dtype=np.int32)
                    })
    else:
        # Run Simulation
        detected_items = get_mock_predictions(image_name, img_w, img_h)
        # Convert simulated polygon coords to np.array
        for item in detected_items:
            item['polygon'] = np.array(item['polygon'], dtype=np.int32)
            
    # Draw segmentation masks and labels
    overlay = img_bgr.copy()
    
    for item in detected_items:
        class_name = item['class']
        mapping = CLASS_MAPPING.get(class_name)
        if not mapping:
            continue
            
        color_rgb = mapping['color_rgb']
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # BGR for OpenCV
        pts = item['polygon']
        
        # Draw transparent mask
        cv2.fillPoly(overlay, [pts], color_bgr)
        # Draw mask border
        cv2.polylines(overlay, [pts], True, color_bgr, 3)
        
    # Blend overlay
    alpha = 0.45
    blended_bgr = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    
    # Add Text Labels
    for item in detected_items:
        class_name = item['class']
        mapping = CLASS_MAPPING.get(class_name)
        if not mapping:
            continue
        color_rgb = mapping['color_rgb']
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        pts = item['polygon']
        
        # Label placement (top-left of polygon)
        ymin_idx = np.argmin([pt[1] for pt in pts])
        text_x = int(pts[ymin_idx][0])
        text_y = int(max(pts[ymin_idx][1] - 8, 20))
        
        # Make text background
        label_text = f"{class_name} ({int(item['confidence']*100)}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(blended_bgr, (text_x, text_y - th - 6), (text_x + tw + 8, text_y + 4), color_bgr, -1)
        cv2.putText(blended_bgr, label_text, (text_x + 4, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Convert BGR back to RGB for display
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    
    # ----------------- DISPLAY RESULTS -----------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("📸 Original Image")
        st.image(img_pil, use_container_width=True)
    with col_img2:
        st.subheader("🎯 Segregated View")
        st.image(blended_rgb, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compute Analytics Metrics
    total_dets = len(detected_items)
    
    if total_dets > 0:
        counts = {'Green Bin': 0, 'Blue Bin': 0, 'Red Bin': 0, 'Black Bin': 0, 'Gray Bin': 0}
        for item in detected_items:
            bin_name = CLASS_MAPPING[item['class']]['bin']
            counts[bin_name] = counts.get(bin_name, 0) + 1
            
        dry_percentage = round((counts['Blue Bin'] / total_dets) * 100)
        wet_percentage = round((counts['Green Bin'] / total_dets) * 100)
        others_percentage = 100 - (dry_percentage + wet_percentage)
        
        # Render Metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        cols_m = st.columns(4)
        with cols_m[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #3B82F6;">{total_dets}</div><div class="metric-label">Trash Detected</div></div>', unsafe_allow_html=True)
        with cols_m[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #10B981;">{wet_percentage}%</div><div class="metric-label">Wet Organic</div></div>', unsafe_allow_html=True)
        with cols_m[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #3b82f6;">{dry_percentage}%</div><div class="metric-label">Dry Recyclable</div></div>', unsafe_allow_html=True)
        with cols_m[3]:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #ef4444;">{others_percentage}%</div><div class="metric-label">Sanitary/Hazardous/C&D</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Render roadmap / list
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Segregation & Disposal Roadmap")
        st.write("Here is the segregation and preparation workflow for each detected item:")
        
        for item in detected_items:
            class_name = item['class']
            mapping = CLASS_MAPPING[class_name]
            
            badge_html = get_bin_badge(mapping['bin'])
            
            # Interactive structured row
            st.markdown(f"""
            <div style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 1.1rem; color: #ffffff;">{class_name}</span>
                    <div>{badge_html}</div>
                </div>
                <div style="margin-top: 5px; font-size: 0.95rem; color: #cbd5e1;">
                    <strong>Category:</strong> {mapping['category']}<br/>
                    <strong>👉 Action Required:</strong> {mapping['instruction']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No trash detected in the image. Please try adjusting confidence thresholds or uploading another image.")
# TAB 1: Live Uploads
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Upload an image of waste items or take a picture using your device camera to classify them.")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with col_up2:
        camera_file = st.camera_input("Capture with Camera")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        process_and_display(img)
    elif camera_file is not None:
        img = Image.open(camera_file)
        process_and_display(img)
# TAB 2: Sandbox examples
with tab2:
    st.write("Select one of the 5 realistic preloaded trash scenarios below to check the segregator results instantly:")
    
    example_images = {
        "scenario1": {"label": "🥤 Plastic Water Bottle", "filename": "plastic_bottle.jpg"},
        "scenario2": {"label": "📦 Cardboard Delivery Box", "filename": "cardboard_box.jpg"},
        "scenario3": {"label": "🥤 Crushed Soda Can", "filename": "metal_can.jpg"},
        "scenario4": {"label": "😷 Surgical Face Mask", "filename": "face_mask.jpg"},
        "scenario5": {"label": "🥥 Broken Coconut Shell", "filename": "coconut_shell.jpg"}
    }
    
    # Render layout grids
    cols = st.columns(5)
    selected_scenario = None
    
    for idx, (key, data) in enumerate(example_images.items()):
        with cols[idx]:
            img_path = os.path.join("examples", data['filename'])
            if os.path.exists(img_path):
                st.image(img_path, caption=data['label'], use_container_width=True)
                if st.button(f"Analyze {data['filename'].split('.')[0].replace('_', ' ').title()}", key=key):
                    selected_scenario = key
            else:
                st.error(f"Missing {data['filename']}")
                
    if selected_scenario:
        data = example_images[selected_scenario]
        img_path = os.path.join("examples", data['filename'])
        img = Image.open(img_path)
        
        st.markdown(f"## Selected: {data['label']}")
        process_and_display(img, image_name=data['filename'])