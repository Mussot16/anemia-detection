# predictor/views.py

import os
import base64
import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render
from .forms import ImageUploadForm
from .predict_anemia import predict_from_rgb

def extract_rgb_from_image(image_path: str):
    """
    Lee la imagen PNG recortada (con fondo blanco) y devuelve
    el promedio de R, G y B SOLO de los píxeles que NO sean
    fondo blanco puro (255,255,255) o casi blanco (R,G,B >= 250).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se puede leer '{image_path}'")

    b, g, r = cv2.split(img)

    umbral = 250
    mask = ~((b >= umbral) & (g >= umbral) & (r >= umbral))

    if not np.any(mask):
        return 0.0, 0.0, 0.0

    b_vals = b[mask].astype(np.float32)
    g_vals = g[mask].astype(np.float32)
    r_vals = r[mask].astype(np.float32)

    avg_r = float(np.mean(r_vals))
    avg_g = float(np.mean(g_vals))
    avg_b = float(np.mean(b_vals))

    return avg_r, avg_g, avg_b


def upload_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST)
        if form.is_valid():
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            b64 = form.cleaned_data["cropped_image_data"]
            header, data = b64.split(",", 1)
            img_data = base64.b64decode(data)

            filename = "input_cropped.png"
            path = os.path.join(settings.MEDIA_ROOT, filename)
            with open(path, "wb") as f:
                f.write(img_data)

            red_avg, green_avg, blue_avg = extract_rgb_from_image(path)

            total = red_avg + green_avg + blue_avg
            if total == 0.0:
                rp, gp, bp = 0.0, 0.0, 0.0
            else:
                rp = round((red_avg   / total) * 100, 2)
                gp = round((green_avg / total) * 100, 2)
                bp = round((blue_avg  / total) * 100, 2)

            sex = int(form.cleaned_data["sex"])

            print(f"DEBUG: sex={sex}, rp={rp}, gp={gp}, bp={bp} → predict_from_rgb({sex}, {rp}, {gp}, {bp})")

            diagnosis, probability = predict_from_rgb(sex, rp, gp, bp)

            return render(request, "result.html", {
                "diagnosis":   diagnosis,
                "probability": round(probability, 4),
                "rgb":         (rp, gp, bp),
                "roi_image":   settings.MEDIA_URL + filename,
            })
    else:
        form = ImageUploadForm()

    return render(request, "upload.html", {"form": form})
