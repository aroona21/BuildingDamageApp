import os
import io
import hashlib
import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageDraw, ImageFont
from model_def import SiamUNet


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Building Damage Assessment",
    page_icon="🏚️",
    layout="wide"
)


# --------------------------------------------------
# 2. DAMAGE CLASSES AND COLOURS
# --------------------------------------------------

CLASS_NAMES = [
    "Background",
    "No damage",
    "Minor damage",
    "Major damage",
    "Destroyed"
]

COLORS = np.array([
    [0, 0, 0],        # Background - Black
    [0, 200, 0],      # No damage - Green
    [255, 255, 0],    # Minor damage - Yellow
    [255, 140, 0],    # Major damage - Orange
    [255, 0, 0]       # Destroyed - Red
], dtype=np.uint8)

CLASS_DESCRIPTIONS = [
    "Unclassified or non-building area",
    "Building remains undamaged",
    "Minor structural damage",
    "Significant structural damage",
    "Building completely destroyed"
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------
# 3. INFERENCE SETTINGS
# --------------------------------------------------

TILE = 512
OVERLAP = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# --------------------------------------------------
# 4. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(checkpoint_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SiamUNet(
        encoder_name="resnet50",
        num_classes=5,
        encoder_weights=None
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, device


# --------------------------------------------------
# 5. TILE POSITION CALCULATION
# --------------------------------------------------

def tile_starts(length, tile, stride):

    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    last = length - tile

    if starts[-1] != last:
        starts.append(last)

    return starts


# --------------------------------------------------
# 6. BUILDING DAMAGE PREDICTION
# --------------------------------------------------

def predict_mask(pre_rgb, post_rgb, model, device):

    if pre_rgb.shape != post_rgb.shape:
        raise ValueError(
            "Pre-disaster and post-disaster images "
            "must have identical dimensions."
        )

    h, w = pre_rgb.shape[:2]

    pad_h = max(0, TILE - h)
    pad_w = max(0, TILE - w)

    if pad_h or pad_w:
        pre_rgb = np.pad(
            pre_rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge"
        )

        post_rgb = np.pad(
            post_rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge"
        )

    H, W = pre_rgb.shape[:2]
    stride = TILE - OVERLAP

    ys = tile_starts(H, TILE, stride)
    xs = tile_starts(W, TILE, stride)

    logits_sum = np.zeros((5, H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    total_tiles = len(ys) * len(xs)
    completed_tiles = 0

    progress = st.progress(0)
    status = st.empty()

    with torch.inference_mode():

        for y in ys:
            for x in xs:

                pre = pre_rgb[
                    y:y + TILE, x:x + TILE
                ].astype(np.float32) / 255.0

                post = post_rgb[
                    y:y + TILE, x:x + TILE
                ].astype(np.float32) / 255.0

                pre = (pre - MEAN) / STD
                post = (post - MEAN) / STD

                pre = torch.from_numpy(
                    pre.transpose(2, 0, 1).copy()
                ).unsqueeze(0).to(device)

                post = torch.from_numpy(
                    post.transpose(2, 0, 1).copy()
                ).unsqueeze(0).to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda")
                ):
                    logits = model(pre, post)

                logits = logits[0].float().cpu().numpy()

                logits_sum[:, y:y + TILE, x:x + TILE] += logits
                weights[y:y + TILE, x:x + TILE] += 1.0

                completed_tiles += 1
                progress.progress(completed_tiles / total_tiles)
                status.text(
                    f"Processing tile {completed_tiles}/{total_tiles}"
                )

    progress.empty()
    status.empty()

    averaged = logits_sum / np.maximum(weights[None, :, :], 1e-6)
    prediction = averaged.argmax(axis=0)

    return prediction[:h, :w].astype(np.uint8)


# --------------------------------------------------
# 7. IMAGE AND DOWNLOAD HELPERS
# --------------------------------------------------

def png_bytes(array, mode=None):

    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def create_overlay(post_image, prediction):

    post_array = np.array(post_image)
    color_mask = COLORS[prediction]
    overlay = post_array.copy()

    visible = prediction > 0

    overlay[visible] = (
        0.5 * post_array[visible].astype(np.float32)
        + 0.5 * color_mask[visible].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return overlay


def create_color_mask(prediction):

    # Class 0 stays black; classes 1-4 use the project colour palette.
    color_mask = np.zeros(
        (prediction.shape[0], prediction.shape[1], 3),
        dtype=np.uint8
    )

    for class_id in range(1, 5):
        color_mask[prediction == class_id] = COLORS[class_id]

    return color_mask


def create_overlay_download(overlay, percentages):

    image = Image.fromarray(overlay).convert("RGB")
    width, height = image.size

    scale = max(1, width / 1000)
    padding = max(12, int(20 * scale))
    header_height = max(35, int(42 * scale))
    row_height = max(30, int(36 * scale))
    square_size = max(14, int(18 * scale))
    font_size = max(12, int(18 * scale))

    legend_height = header_height + row_height * 4 + padding * 2

    final_image = Image.new(
        "RGB",
        (width, height + legend_height),
        "white"
    )

    final_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(final_image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    draw.rectangle(
        (0, height, width, height + legend_height),
        fill=(255, 255, 255)
    )

    draw.text(
        (padding, height + padding),
        "DAMAGE DISTRIBUTION",
        fill=(20, 20, 20),
        font=font
    )

    for index, class_id in enumerate(range(1, 5)):

        y = height + padding + header_height + index * row_height

        color = tuple(int(value) for value in COLORS[class_id])

        draw.rectangle(
            (
                padding,
                y,
                padding + square_size,
                y + square_size
            ),
            fill=color,
            outline=(70, 70, 70)
        )

        draw.text(
            (padding + square_size + padding // 2, y),
            CLASS_NAMES[class_id],
            fill=(20, 20, 20),
            font=font
        )

        percentage_text = f"{percentages[class_id]:.2f}%"

        draw.text(
            (width - padding - 100, y),
            percentage_text,
            fill=(20, 20, 20),
            font=font
        )

    buffer = io.BytesIO()
    final_image.save(buffer, format="PNG")

    return buffer.getvalue()


def render_damage_bar(class_id, percentage):
    """
    Display a custom class-coloured bar.
    Very small non-zero values receive a 1.2% minimum visual width
    so they remain visible; the numeric percentage stays exact.
    """

    color = "#{:02x}{:02x}{:02x}".format(
        int(COLORS[class_id][0]),
        int(COLORS[class_id][1]),
        int(COLORS[class_id][2])
    )

    visual_percentage = min(
        100.0,
        max(float(percentage), 1.2) if percentage > 0 else 0.0
    )

    minimum_width = "4px" if percentage > 0 else "0"

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin:13px 0 7px 0;
        ">
            <div style="display:flex;align-items:center;gap:9px;">
                <span style="
                    display:inline-block;
                    width:13px;
                    height:13px;
                    border-radius:4px;
                    background:{color};
                "></span>
                <span style="font-weight:600;">
                    {CLASS_NAMES[class_id]}
                </span>
            </div>
            <span style="
                font-weight:600;
                font-variant-numeric:tabular-nums;
            ">{percentage:.2f}%</span>
        </div>

        <div style="
            width:100%;
            height:9px;
            background:rgba(128,128,128,0.22);
            border-radius:20px;
            overflow:hidden;
            margin-bottom:8px;
        ">
            <div style="
                width:{visual_percentage:.3f}%;
                min-width:{minimum_width};
                height:100%;
                background:{color};
                border-radius:20px;
            "></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_image_fingerprint(pre_image, post_image):

    image_hash = hashlib.sha256()
    image_hash.update(pre_image.tobytes())
    image_hash.update(post_image.tobytes())

    return image_hash.hexdigest()


# --------------------------------------------------
# 8. SIDEBAR - TERRA SCAN
# --------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 10px 4px 14px 4px;">
        <h2 style="font-size:23px; margin-bottom:4px;">TerraScan</h2>
        <p style="font-size:11px; color:#9ca3af; margin-top:0; letter-spacing:1.2px;">
            BUILDING DAMAGE ASSESSMENT
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.markdown("#### Display Settings")

display_scale = st.sidebar.slider(
    "Image display scale (%)",
    min_value=40,
    max_value=100,
    value=80,
    step=10,
    format="%d%%"
)

display_width = int(900 * display_scale / 100)

st.sidebar.divider()
st.sidebar.markdown("#### Damage Classification")
st.sidebar.caption("Colour indicators used in the predicted damage masks.")

for i, class_name in enumerate(CLASS_NAMES):

    color = COLORS[i]
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(color[0]), int(color[1]), int(color[2])
    )

    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; margin-bottom:12px;">
            <div style="
                width:15px; height:15px; min-width:15px;
                background-color:{hex_color}; border-radius:4px;
                margin-right:11px;
                border:1px solid rgba(255,255,255,0.3);
            "></div>
            <div>
                <div style="font-size:13px; font-weight:600;">{class_name}</div>
                <div style="font-size:10px; color:#9ca3af; margin-top:2px;">
                    {CLASS_DESCRIPTIONS[i]}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style="font-size:12px; line-height:1.6;">
        <div style="font-weight:600;">Satellite Image Analysis</div>
        <div style="color:#9ca3af;">AI-Based Damage Classification</div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 9. MAIN APPLICATION INTERFACE
# --------------------------------------------------

st.title("Building Damage Assessment")

st.caption(
    "Pre- and post-disaster satellite image analysis"
)

checkpoint_path = os.path.join(
    os.path.dirname(__file__),
    "checkpoints",
    "resnet50_damage_best.pt"
)

if not os.path.exists(checkpoint_path):

    st.error(
        "Model checkpoint not found. "
        "Place resnet50_damage_best.pt inside "
        "the checkpoints folder."
    )

    st.stop()

if DEVICE.type == "cuda":
    st.caption("Processing device: GPU")
else:
    st.caption("Processing device: CPU")


# --------------------------------------------------
# 10. IMAGE UPLOAD SECTION
# --------------------------------------------------

st.subheader("Upload Satellite Images")

col1, col2 = st.columns(2)

with col1:
    pre_file = st.file_uploader(
        "Pre-disaster image",
        type=["png", "jpg", "jpeg"],
        key="pre"
    )

with col2:
    post_file = st.file_uploader(
        "Post-disaster image",
        type=["png", "jpg", "jpeg"],
        key="post"
    )


# --------------------------------------------------
# 11. IMAGE PREVIEW
# --------------------------------------------------

if pre_file is not None and post_file is not None:

    pre_image = Image.open(pre_file).convert("RGB")
    post_image = Image.open(post_file).convert("RGB")

    if pre_image.size != post_image.size:

        st.error(
            f"Image dimensions must match. "
            f"Pre-disaster: {pre_image.size}; "
            f"Post-disaster: {post_image.size}."
        )

        st.stop()

    st.subheader("Uploaded Images")

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:
        st.image(
            pre_image,
            caption="Pre-disaster Image",
            width=display_width
        )

    with preview_col2:
        st.image(
            post_image,
            caption="Post-disaster Image",
            width=display_width
        )


    # --------------------------------------------------
    # 12. ANALYZE BUTTON
    # --------------------------------------------------

    if st.button(
        "Analyze Building Damage",
        type="primary",
        use_container_width=True
    ):

        try:

            model, device = load_model(checkpoint_path)

            with st.spinner("Analyzing satellite images..."):

                prediction = predict_mask(
                    np.array(pre_image),
                    np.array(post_image),
                    model,
                    device
                )

                overlay = create_overlay(
                    post_image,
                    prediction
                )

                color_mask = create_color_mask(prediction)

                # Calculate pixel-area distribution.
                total_pixels = prediction.size

                counts = np.bincount(
                    prediction.ravel(),
                    minlength=5
                )

                percentages = {}

                for class_id in range(1, 5):

                    percentage = (
                        100.0 * counts[class_id]
                        / max(total_pixels, 1)
                    )

                    percentages[class_id] = percentage

                fingerprint = get_image_fingerprint(
                    pre_image,
                    post_image
                )

                st.session_state["damage_result"] = {
                    "fingerprint": fingerprint,
                    "prediction": prediction,
                    "overlay": overlay,
                    "color_mask": color_mask,
                    "percentages": percentages
                }

            st.success("Building damage analysis completed!")

        except Exception as exc:

            st.error("An error occurred during analysis.")
            st.exception(exc)


    # --------------------------------------------------
    # 13. DISPLAY RESULTS
    # --------------------------------------------------

    fingerprint = get_image_fingerprint(
        pre_image,
        post_image
    )

    result = st.session_state.get("damage_result")

    if (
        result is not None
        and result["fingerprint"] == fingerprint
    ):

        st.divider()

        st.header("Damage Assessment Results")

        st.caption(
            "Compare the predicted damage overlay with the "
            "colour-coded overlay and its damage percentages."
        )

        damage_overlay_bytes = create_overlay_download(
            result["overlay"],
            result["percentages"]
        )

        # Display both result images side by side.
        result_col1, result_col2 = st.columns(
            2,
            gap="large"
        )

        with result_col1:

            st.markdown("#### Predicted Damage Mask")

            st.image(
                result["color_mask"],
                caption="Black background with colour-coded predicted damage classes",
                use_container_width=True
            )

        with result_col2:

            st.markdown("#### Damage Overlay")

            st.image(
                result["overlay"],
                caption="Predicted damage areas highlighted on the post-disaster image",
                use_container_width=True
            )


        # --------------------------------------------------
        # 15. DAMAGE DISTRIBUTION
        # --------------------------------------------------

        st.divider()

        st.subheader("Predicted Damage Distribution")

        st.caption(
            "Percentages show the proportion of image pixels "
            "assigned to each damage class. The bar colours "
            "match the damage classification legend."
        )

        for class_id in range(1, 5):

            render_damage_bar(
                class_id,
                result["percentages"][class_id]
            )


        # --------------------------------------------------
        # 16. DOWNLOAD RESULTS
        # --------------------------------------------------

        st.divider()

        st.subheader("Download Results")

        download_col1, download_col2, download_col3 = st.columns(3)

        with download_col1:

            st.download_button(
                label="Download Damage Mask",
                data=png_bytes(
                    result["color_mask"],
                    mode="RGB"
                ),
                file_name="predicted_damage_mask.png",
                mime="image/png",
                use_container_width=True
            )

        with download_col2:

            st.download_button(
                label="Download Predicted Overlay",
                data=png_bytes(
                    result["overlay"],
                    mode="RGB"
                ),
                file_name="predicted_overlay.png",
                mime="image/png",
                use_container_width=True
            )

        with download_col3:

            st.download_button(
                label="Download Damage Overlay",
                data=damage_overlay_bytes,
                file_name="damage_overlay_with_percentages.png",
                mime="image/png",
                use_container_width=True
            )

        st.caption(
            "Note: Damage percentages represent pixel-area "
            "distribution, not the exact number of damaged "
            "buildings. Predictions should be reviewed "
            "before being used for decision-making."
        )

else:

    st.info(
        "Please upload both pre-disaster and "
        "post-disaster satellite images to begin."
    )
