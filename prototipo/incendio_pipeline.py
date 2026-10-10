#!/usr/bin/env python3
"""
incendio_pipeline.py
====================
Detección y caracterización de un incendio forestal con dos composiciones
Sentinel-2 de la MISMA escena: True Color (TC) y SWIR.

Qué hace (en orden):
  1. Carga las dos imágenes y descarta banner / créditos / logos.
  2. Segmenta  -> zona quemada (SWIR), humo (TC) y frente activo (SWIR).
  3. Mide      -> hectáreas, perímetro, centroide, caja, dirección del humo...
  4. Dibuja    -> overlays sobre TC y SWIR + zoom con flechas.
  5. Exporta   -> metricas.json, máscaras PNG y GeoJSON.

Uso:
    python incendio_pipeline.py tc.jpg swir.jpg --out resultados
    python incendio_pipeline.py tc.jpg swir.jpg --out resultados --px-per-km 7.6
    python incendio_pipeline.py tc.tif swir.tif --out resultados --bounds -5.7 39.5 -2.0 41.0

Notas:
  * Escala: con "--px-per-km auto" (por defecto) intenta leer la barra de escala
    de Copernicus Browser (10 km). Si tienes un GeoTIFF, pasa la escala real.
  * --bounds W S E N (grados) georreferencia el GeoJSON suponiendo imagen
    norte-arriba en Web Mercator (como exporta Copernicus Browser).
    Sin --bounds, el GeoJSON sale en coordenadas de píxel.
  * Todos los umbrales están en PARAMS: es lo primero que hay que ajustar.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# PARÁMETROS (ajustar aquí)
# --------------------------------------------------------------------------
PARAMS = dict(
    # recorte de elementos de la interfaz (fracciones del alto/ancho)
    margin_top_px=50,            # banner con la fecha
    margin_bottom_px=32,         # línea de créditos
    ignore_boxes=[(0.93, 0.92, 1.0, 1.0),    # logo Copernicus (x0,y0,x1,y1 en fracción)
                  (0.0, 0.92, 0.10, 1.0)],   # barra de escala

    # ZONA QUEMADA (imagen SWIR, HSV de OpenCV: H 0-180)
    burn_h_max=16, burn_h_min=170,   # tono rojizo / marrón
    burn_s_min=100,                  # saturado
    burn_v_min=45, burn_v_max=215,   # ni negro (agua) ni muy claro (etiquetas)
    burn_rg_min=25,                  # R - G mínimo (quemado: rojo > verde)
    # regla 2: quemado oscuro/oliva (R ~ G bajos y B muy bajo; la vegetación sana tiene G alto)
    burn2_g_max=115, burn2_b_max=80, burn2_s_min=120, burn2_rg_min=-8,
    burn_min_ha=40,                  # descarta manchas más pequeñas

    # HUMO (imagen True Color): gris/azulado y claro; el suelo es cálido (R > B)
    smoke_blur=3,                    # suavizado previo (sigma, px)
    smoke_rb_max=9,                  # R - B máximo (humo ~ neutro o azulado)
    smoke_s_max=30,                  # saturación máxima
    smoke_v_min=140,                 # brillo mínimo
    smoke_min_ha=3000,               # descarta manchas pequeñas (ciudades, nubes sueltas)
    smoke_attach_km=15,              # el humo debe estar pegado a una cicatriz

    # FRENTE ACTIVO (imagen SWIR): naranja/amarillo muy brillante
    front_h_min=5, front_h_max=28,
    front_s_min=170, front_v_min=215,
    front_min_px=15,                 # si hay menos píxeles, se considera que no hay frente visible
    front_infer_km=3,                # frente inferido: borde de la cicatriz a <= X km del humo
)


# --------------------------------------------------------------------------
# 1. CARGA
# --------------------------------------------------------------------------
def load_pair(tc_path, swir_path):
    tc = cv2.imread(tc_path, cv2.IMREAD_COLOR)
    sw = cv2.imread(swir_path, cv2.IMREAD_COLOR)
    if tc is None or sw is None:
        raise SystemExit("No se pudo leer alguna imagen (¿ruta correcta?)")
    tc = cv2.cvtColor(tc, cv2.COLOR_BGR2RGB)
    sw = cv2.cvtColor(sw, cv2.COLOR_BGR2RGB)
    if tc.shape != sw.shape:
        print(f"[aviso] tamaños distintos {tc.shape} vs {sw.shape}: redimensiono SWIR al de TC")
        sw = cv2.resize(sw, (tc.shape[1], tc.shape[0]), interpolation=cv2.INTER_AREA)
    return tc, sw


def valid_mask(shape, p):
    h, w = shape[:2]
    m = np.ones((h, w), bool)
    m[: p["margin_top_px"], :] = False
    m[h - p["margin_bottom_px"]:, :] = False
    for x0, y0, x1, y1 in p["ignore_boxes"]:
        m[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)] = False
    return m


def detect_scalebar_px_per_km(img, km=10.0):
    """Busca la barra blanca de escala (abajo a la izquierda) y devuelve px/km."""
    h, w = img.shape[:2]
    roi = img[int(0.90 * h):, : int(0.12 * w)]
    white = roi.min(axis=2) > 235
    best = 0
    for row in white:
        # longitud de la racha blanca más larga de la fila
        run = longest = 0
        for v in row:
            run = run + 1 if v else 0
            longest = max(longest, run)
        best = max(best, longest)
    # la barra mide ~50-110 px; si salen valores absurdos, no nos fiamos
    if 30 <= best <= 300:
        return best / km
    return None


# --------------------------------------------------------------------------
# 2. SEGMENTACIÓN
# --------------------------------------------------------------------------
def clean_mask(m, close=7, open_=3, min_px=0):
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_, open_))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k_c)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_o)
    m = ndi.binary_fill_holes(m)
    if min_px > 0:
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            keep = np.isin(lab, 1 + np.where(sizes >= min_px)[0])
            m = keep
    return m.astype(bool)


def segment_burned(sw, valid, p, ha_per_px):
    hsv = cv2.cvtColor(sw, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    r, g, b = [sw[..., i].astype(int) for i in range(3)]
    rojo = (((h <= p["burn_h_max"]) | (h >= p["burn_h_min"]))
            & (s >= p["burn_s_min"]) & (v >= p["burn_v_min"]) & (v <= p["burn_v_max"])
            & (r - g >= p["burn_rg_min"]))
    oscuro = ((g <= p["burn2_g_max"]) & (b <= p["burn2_b_max"]) & (s >= p["burn2_s_min"])
              & (r - g >= p["burn2_rg_min"]) & (v >= p["burn_v_min"]))
    m = (rojo | oscuro) & valid
    return clean_mask(m, close=7, open_=3, min_px=int(p["burn_min_ha"] / ha_per_px))


def segment_smoke_candidates(tc, valid, p, ha_per_px):
    t = cv2.GaussianBlur(tc, (0, 0), p["smoke_blur"])
    hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    rb = t[..., 0].astype(int) - t[..., 2].astype(int)
    m = (rb <= p["smoke_rb_max"]) & (s <= p["smoke_s_max"]) & (v >= p["smoke_v_min"]) & valid
    return clean_mask(m, close=15, open_=7, min_px=int(p["smoke_min_ha"] / ha_per_px))


def segment_front(sw, valid, p):
    hsv = cv2.cvtColor(sw, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    m = ((h >= p["front_h_min"]) & (h <= p["front_h_max"])
         & (s >= p["front_s_min"]) & (v >= p["front_v_min"]) & valid)
    return m


def split_components(mask):
    lab, n = ndi.label(mask)
    return lab, n


# --------------------------------------------------------------------------
# 3. MÉTRICAS
# --------------------------------------------------------------------------
def bearing_deg(dx, dy_img):
    """Rumbo (0=N, 90=E) para un desplazamiento en imagen (x→este, y→sur)."""
    return (math.degrees(math.atan2(dx, -dy_img)) + 360) % 360


def compass(b):
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return pts[int((b + 11.25) // 22.5) % 16]


def region_metrics(mask, km_per_px):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim_px = sum(cv2.arcLength(cv2.approxPolyDP(c, 3.0, True), True) for c in cnts)
    area_km2 = xs.size * km_per_px ** 2
    return dict(
        area_ha=round(area_km2 * 100, 1),
        area_km2=round(area_km2, 2),
        perimetro_km=round(perim_px * km_per_px, 1),
        centroide_px=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        bbox_px=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    )


def plume_axis(smoke, origin_xy, km_per_px):
    """PCA de la máscara de humo -> eje, largo, ancho y rumbo (origen -> extremo lejano)."""
    ys, xs = np.nonzero(smoke)
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, vec = np.linalg.eigh(cov)
    ax = vec[:, 1]                       # eje principal
    perp = vec[:, 0]
    t = (pts - c) @ ax
    u = (pts - c) @ perp
    t0, t1 = np.percentile(t, [1, 99])
    u0, u1 = np.percentile(u, [1, 99])
    p_a, p_b = c + ax * t0, c + ax * t1  # extremos
    # el origen del penacho es el extremo más cercano a la cicatriz
    if np.linalg.norm(p_a - origin_xy) <= np.linalg.norm(p_b - origin_xy):
        start, end = p_a, p_b
    else:
        start, end = p_b, p_a
    d = end - start
    return dict(start=start, end=end,
                largo_km=round(float(np.hypot(*d)) * km_per_px, 1),
                ancho_km=round(float(u1 - u0) * km_per_px, 1),
                rumbo_hacia_deg=round(bearing_deg(d[0], d[1]), 0))


# --------------------------------------------------------------------------
# 4. DIBUJO
# --------------------------------------------------------------------------
COL = dict(main=(255, 215, 0), other=(255, 130, 0), smoke=(0, 210, 255), front=(255, 0, 60))


def draw_overlay(img, layers, alpha=0.30, thick=2):
    out = img.copy()
    for mask, color in layers:
        if mask is None or not mask.any():
            continue
        fill = np.zeros_like(out)
        fill[:] = color
        out = np.where(mask[..., None], (out * (1 - alpha) + fill * alpha).astype(np.uint8), out)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, thick)
    return out


def legend_handles(has_front, has_other):
    f = lambda c: tuple(v / 255 for v in c)
    h = [Patch(color=f(COL["main"]), label="Zona quemada (incendio principal)"),
         Patch(color=f(COL["smoke"]), label="Humo")]
    if has_front:
        h.append(Patch(color=f(COL["front"]), label="Frente activo / frente probable"))
    if has_other:
        h.append(Patch(color=f(COL["other"]), label="Otras cicatrices"))
    return h


def save_overlay_png(img, path, title, handles):
    fig = plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150 + 0.4), dpi=150)
    ax = fig.add_axes([0, 0, 1, 0.94])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.01, 0.965, title, fontsize=12, weight="bold", va="center")
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.99, 0.995))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_zoom(tc, sw, layers, info, path, px_per_km):
    x0, y0, x1, y1 = info["zoom_bbox"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.4), dpi=130)
    for ax, base, name in zip(axs, (tc, sw), ("True Color", "SWIR")):
        ax.imshow(draw_overlay(base, layers, alpha=0.22, thick=2)[y0:y1, x0:x1],
                  extent=(x0, x1, y1, y0))
        ax.set_title(name, fontsize=12, weight="bold")
        ax.axis("off")
        if info.get("spread"):
            (sx, sy), (ex, ey) = info["spread"]
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=4, mutation_scale=22))
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color="#ff003c", lw=2.2, mutation_scale=22))
        if info.get("plume"):
            (sx, sy), (ex, ey) = info["plume"]
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=4, mutation_scale=22))
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color="#00b4d8", lw=2.2, mutation_scale=22))
        # barra de escala de 10 km
        bx, by = x0 + 25, y1 - 25
        ax.plot([bx, bx + 10 * px_per_km], [by, by], color="white", lw=5, solid_capstyle="butt")
        ax.plot([bx, bx + 10 * px_per_km], [by, by], color="black", lw=2, solid_capstyle="butt")
        ax.text(bx + 5 * px_per_km, by - 10, "10 km", color="white", ha="center", fontsize=9,
                weight="bold", bbox=dict(facecolor="black", alpha=0.45, pad=1.5, lw=0))
        ax.annotate("N", xy=(x1 - 40, y0 + 70), xytext=(x1 - 40, y0 + 120), ha="center", color="white",
                    fontsize=12, weight="bold", arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    fig.suptitle(info["titulo"], fontsize=14, weight="bold", y=0.98)
    fig.legend(handles=info["handles"], loc="upper center", ncol=len(info["handles"]), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.text(0.5, 0.03, info["pie"], ha="center", fontsize=11, weight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.03)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. EXPORTACIÓN
# --------------------------------------------------------------------------
def make_px2geo(bounds, shape):
    """Píxel -> (lon, lat). Si no hay bounds, devuelve el propio píxel."""
    if bounds is None:
        return lambda x, y: [float(x), float(y)]
    w_, s_, e_, n_ = bounds
    h, w = shape[:2]
    merc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    inv = lambda m: math.degrees(2 * math.atan(math.exp(m)) - math.pi / 2)
    ms, mn = merc(s_), merc(n_)

    def f(x, y):
        lon = w_ + (e_ - w_) * x / w
        lat = inv(mn - (mn - ms) * y / h)
        return [round(lon, 6), round(lat, 6)]
    return f


def mask_to_features(mask, name, px2geo, props=None, simplify=2.0):
    feats = []
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue
        c = cv2.approxPolyDP(c, simplify, True)[:, 0, :]
        if len(c) < 3:
            continue
        ring = [px2geo(x, y) for x, y in c] + [px2geo(*c[0])]
        feats.append(dict(type="Feature",
                          properties=dict(clase=name, **(props or {})),
                          geometry=dict(type="Polygon", coordinates=[ring])))
    return feats


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tc")
    ap.add_argument("swir")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--px-per-km", default="auto")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    a = ap.parse_args()
    p = PARAMS
    os.makedirs(a.out, exist_ok=True)

    tc, sw = load_pair(a.tc, a.swir)
    H, W = tc.shape[:2]

    # --- escala
    if a.px_per_km == "auto":
        ppk = detect_scalebar_px_per_km(tc) or detect_scalebar_px_per_km(sw)
        if ppk is None:
            raise SystemExit("No encuentro la barra de escala: pasa --px-per-km X")
        print(f"[escala] barra de 10 km detectada -> {ppk:.2f} px/km ({1000 / ppk:.0f} m/px)")
    else:
        ppk = float(a.px_per_km)
    km_per_px = 1 / ppk
    ha_per_px = (km_per_px ** 2) * 100

    valid = valid_mask(tc.shape, p)

    # --- 1) quemado, humo candidato
    burned_all = segment_burned(sw, valid, p, ha_per_px)
    smoke_cand = segment_smoke_candidates(tc, valid, p, ha_per_px)

    # --- 2) incendio principal = cicatrices pegadas al humo
    attach_px = p["smoke_attach_km"] * ppk
    if burned_all.any() and smoke_cand.any():
        dist_burn = ndi.distance_transform_edt(~burned_all)
        lab_s, ns = ndi.label(smoke_cand)
        keep_s = [i for i in range(1, ns + 1) if dist_burn[lab_s == i].min() <= attach_px]
        plume = np.isin(lab_s, keep_s)
    else:
        plume = np.zeros_like(burned_all)

    lab_b, nb = ndi.label(burned_all)
    if plume.any():
        dist_plume = ndi.distance_transform_edt(~plume)
        keep_b = [i for i in range(1, nb + 1) if dist_plume[lab_b == i].min() <= attach_px]
    else:
        keep_b = list(range(1, nb + 1))
    main_burn = np.isin(lab_b, keep_b)
    other_burn = burned_all & ~main_burn

    # --- 3) frente activo (SWIR) o inferido
    front_raw = segment_front(sw, valid, p)
    front_raw &= ndi.binary_dilation(main_burn, iterations=int(2 * ppk)) if main_burn.any() else False
    if front_raw.sum() >= p["front_min_px"]:
        front, front_kind = front_raw, "detectado (píxeles naranja/amarillo en SWIR)"
    elif main_burn.any() and plume.any():
        filled = ndi.binary_fill_holes(ndi.binary_closing(main_burn, iterations=3))
        edge = filled & ~ndi.binary_erosion(filled, iterations=2)
        near = ndi.distance_transform_edt(~plume) <= p["front_infer_km"] * ppk
        front = edge & near
        front_kind = "inferido (borde de la cicatriz pegado al humo; no hay píxeles de fuego saturados)"
    else:
        front, front_kind = np.zeros_like(main_burn), "no detectado"
    front_vis = ndi.binary_dilation(front, iterations=2)

    # --- 4) métricas
    out = dict(
        escala=dict(px_per_km=round(ppk, 2), m_por_px=round(1000 / ppk, 0), imagen_px=[W, H]),
        parametros_usados={k: v for k, v in p.items() if k != "ignore_boxes"},
    )
    out["incendio_principal"] = region_metrics(main_burn, km_per_px)
    comps = []
    lab_all, n_all = ndi.label(burned_all)
    for i in range(1, n_all + 1):
        m = region_metrics(lab_all == i, km_per_px)
        m["id"] = i
        m["principal"] = bool(main_burn[lab_all == i].any())
        comps.append(m)
    comps.sort(key=lambda d: -d["area_ha"])
    out["cicatrices"] = comps
    out["humo"] = region_metrics(plume, km_per_px)
    out["frente"] = dict(tipo=front_kind, pixeles=int(front.sum()))

    info = dict(spread=None, plume=None)
    if main_burn.any():
        ys, xs = np.nonzero(main_burn)
        c_burn = np.array([xs.mean(), ys.mean()])
        if front.any():
            fy, fx = np.nonzero(front)
            c_front = np.array([fx.mean(), fy.mean()])
            d = c_front - c_burn
            b = bearing_deg(d[0], d[1])
            out["propagacion"] = dict(rumbo_deg=round(b, 0), direccion=compass(b),
                                      nota="vector centroide cicatriz -> frente (orientativo)")
            info["spread"] = (c_burn, c_front)
        if plume.any():
            pa = plume_axis(plume, c_burn, km_per_px)
            info["plume"] = (pa["start"], pa["end"])
            b = pa["rumbo_hacia_deg"]
            out["penacho"] = dict(largo_km=pa["largo_km"], ancho_km=pa["ancho_km"],
                                  rumbo_hacia_deg=b, va_hacia=compass(b),
                                  viento_probable_del=compass((b + 180) % 360),
                                  nota="el humo se mueve con el viento: contrastar con ERA5/AEMET")

    # --- 5) figuras
    layers = [(other_burn, COL["other"]), (main_burn, COL["main"]), (plume, COL["smoke"]), (front_vis, COL["front"])]
    handles = legend_handles(front.any(), other_burn.any())
    save_overlay_png(draw_overlay(tc, layers), os.path.join(a.out, "01_overlay_truecolor.png"),
                     "True Color + segmentación", handles)
    save_overlay_png(draw_overlay(sw, layers), os.path.join(a.out, "02_overlay_swir.png"),
                     "SWIR + segmentación", handles)

    zone = main_burn | plume
    if zone.any():
        ys, xs = np.nonzero(zone)
        m_ = int(8 * ppk)
        info["zoom_bbox"] = (max(0, xs.min() - m_), max(0, ys.min() - m_), min(W, xs.max() + m_), min(H, ys.max() + m_))
        ip, pe = out["incendio_principal"], out.get("penacho")
        info["titulo"] = "Incendio principal: caracterización"
        parts = []
        if ip:
            parts.append(f"Quemado: {ip['area_ha']:,.0f} ha ({ip['area_km2']:.0f} km²) · perímetro {ip['perimetro_km']:.0f} km")
        if pe:
            parts.append(f"Humo hacia el {pe['va_hacia']} ({pe['rumbo_hacia_deg']:.0f}°) · {pe['largo_km']:.0f} km de largo")
        info["pie"] = "   |   ".join(parts)
        info["handles"] = handles
        save_zoom(tc, sw, layers, info, os.path.join(a.out, "03_zoom_caracterizacion.png"), ppk)

    # --- 6) máscaras y GeoJSON
    for name, m in (("quemado", burned_all), ("humo", plume), ("frente", front_vis)):
        cv2.imwrite(os.path.join(a.out, f"mask_{name}.png"), (m * 255).astype(np.uint8))

    px2geo = make_px2geo(a.bounds, tc.shape)
    feats = []
    for i in range(1, n_all + 1):
        mm = lab_all == i
        feats += mask_to_features(mm, "zona_quemada", px2geo,
                                  dict(id=i, principal=bool(main_burn[mm].any()),
                                       area_ha=round(float(mm.sum() * ha_per_px), 1)))
    feats += mask_to_features(plume, "humo", px2geo, dict(area_ha=round(float(plume.sum() * ha_per_px), 1)))
    feats += mask_to_features(front_vis, "frente", px2geo, dict(tipo=front_kind))
    gj = dict(type="FeatureCollection", features=feats,
              crs_note="WGS84 lon/lat (aprox., desde --bounds)" if a.bounds else "COORDENADAS DE PÍXEL (sin --bounds)")
    with open(os.path.join(a.out, "incendio.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)

    with open(os.path.join(a.out, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # --- resumen en consola
    print("\n=== RESUMEN ===")
    if out["incendio_principal"]:
        ip = out["incendio_principal"]
        print(f"Incendio principal: {ip['area_ha']:,.0f} ha | perímetro {ip['perimetro_km']} km | centroide px {ip['centroide_px']}")
    print(f"Otras cicatrices: {sum(1 for c in comps if not c['principal'])}")
    if out["humo"]:
        print(f"Humo: {out['humo']['area_km2']:,.0f} km²")
    print(f"Frente: {front_kind}")
    if "penacho" in out:
        pe = out["penacho"]
        print(f"Penacho: {pe['largo_km']} km x {pe['ancho_km']} km, va hacia el {pe['va_hacia']} ({pe['rumbo_hacia_deg']:.0f}°)")
    print(f"Resultados en: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
