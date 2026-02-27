import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.ops import linemerge
from shapely.geometry import LineString, Polygon
import rasterio
from rasterstats import zonal_stats

# Import untuk ReportLab (PDF)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import pagesizes
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(page_title="GeoPCI System", page_icon="🛣️", layout="wide")

st.title("🛣️ GeoPCI: Sistem Analisis Kondisi Perkerasan Jalan")
st.markdown("Otomatisasi perhitungan Pavement Condition Index (PCI) metode ASTM D6433.")

st.divider()

# =========================================
# INISIALISASI MEMORI (SESSION STATE)
# =========================================
if 'proses_selesai' not in st.session_state:
    st.session_state.proses_selesai = False
if 'df_pci' not in st.session_state:
    st.session_state.df_pci = None
if 'df_detail' not in st.session_state:
    st.session_state.df_detail = None
if 'peta_bytes' not in st.session_state:
    st.session_state.peta_bytes = None
if 'grafik_bytes' not in st.session_state:
    st.session_state.grafik_bytes = None
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None

# ==========================================================
# DATABASE KURVA PCI (STANDAR POLINOMIAL ASTM)
# ==========================================================
DISTRESS_COEFFICIENTS = {
    "alligator_cracking": {
        "valid_min": 0.1, "valid_max": 100.0, "chart_type": "log",
        "coefficients": {
            "low": [11.81030706543641, 14.716555458137659, 5.254969146645571],
            "medium": [21.641468980402742, 19.850106754347717, 4.129291159183641],
            "high": [30.698348853111792, 26.819142548270502, 5.653897800825902, -2.0562458038186975]
        }
    },
    "bleeding": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [0.322105531453021, -0.174525478036764, 1.504981533364469, 1.7947851128512355], "medium": [3.3241213258799234, 4.4914393599717854, 3.3913394399693146, 1.7791635264801178], "high": [5.739963736596728, 7.319479502282341, 7.086578195546586, 3.075398183100246]}},
    "block_cracking": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [-0.24506618985589634, 3.5865225127793514, 4.864809264954378], "medium": [2.0044867717721253, 7.656841349922775, 6.197308536458376], "high": [6.054836363905478, 14.23908143225682, 9.26078972426816]}},
    "bumps_and_sags": {"valid_min": 0.1, "valid_max": 10.0, "chart_type": "log", "coefficients": {"low": [7.432162927047138, 13.039704473951343, 12.681378093050206, 5.724727901683913], "medium": [24.0510616920146, 24.990801854094467, 17.775330890078287, 10.801916366371941], "high": [53.60256635094115, 38.448582442650476, 6.043250542072165]}},
    "corrugation": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [2.079645270606063, 6.171432244797827, 6.315591279321682], "medium": [16.00629930412681, 17.42721794385406, 5.877793175640937], "high": [33.3926340071105, 25.16870941731629, 2.8124490453717783]}},
    "depression": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [3.289661412212115, 0.6252836980242593, 10.964451357494776, 6.985101871322738, -3.5060456381316634], "medium": [7.614400618717246, 3.769091495110122, 15.559272647446328, 7.4759184063961435, -4.844857281255507], "high": [15.948684201140967, 9.394030228694518, 15.510038701741792, 6.024061067798476, -4.422031207577385]}},
    "edge_cracking": {"valid_min": 0.1, "valid_max": 20.0, "chart_type": "log", "coefficients": {"low": [3.211409886900335, 4.962171357579026, 2.4025247047022504], "medium": [9.38292671744566, 9.608124129013511, 4.3396040463768015], "high": [15.664668848033116, 15.377839697997818, 7.293639747414968]}},
    "joint_reflection_cracking": {"valid_min": 0.1, "valid_max": 30.0, "chart_type": "log", "coefficients": {"low": [2.599637864741199, 7.548320214895041, 6.035620574755107], "medium": [7.381829148226155, 13.409513965059839, 14.487037415701224, 2.604980254156252, -4.850878879939305], "high": [15.708313735606989, 21.751135896400438, 22.35224402708013, 12.321810537440612, -6.404620335343928, -4.635950974262599]}},
    "lane_shoulder_drop_off": {"valid_min": 0.5, "valid_max": 15.0, "chart_type": "log", "coefficients": {"low": [1.7330434661503789, 2.2479876898054147, 8.492414136167602], "medium": [3.1018055744999637, 0.9748217115112947, 15.880585262031168], "high": [5.588313880268777, 5.140183474398842, 22.853461108687]}},
    "longitudinal_transverse_cracking": {"valid_min": 0.1, "valid_max": 30.0, "chart_type": "log", "coefficients": {"low": [2.1831950830355784, 8.216281290902756, 6.454535216997481], "medium": [8.67907975505781, 15.31510859022643, 6.266899017617167], "high": [17.723256518500968, 24.494224894065376, 19.119102009732657, 4.182799471044728, -4.54528270804704]}},
    "patching_and_utility_cut_patching": {"valid_min": 0.1, "valid_max": 50.0, "chart_type": "log", "coefficients": {"low": [1.8384973325128793, 7.632382033226324, 6.507032451977727], "medium": [8.929258996532853, 14.056348057074333, 8.632199015982204], "high": [18.05379050800971, 18.570848940912526, 15.195881745184764, 4.634730303593308, -4.282412070934171]}},
    "polished_aggregate": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [0.16789523734739448, -0.1540196727901293, 1.3119789947979732, 1.8893029190226103], "medium": [0.16789523734739448, -0.1540196727901293, 1.3119789947979732, 1.8893029190226103], "high": [0.16789523734739448, -0.1540196727901293, 1.3119789947979732, 1.8893029190226103]}},
    "potholes": {"valid_min": 0.01, "valid_max": 10.0, "chart_type": "log", "coefficients": {"low": [58.574736886209024, 41.33443795417745, 2.307796828048822, -2.1009955078236295], "medium": [91.64010353001909, 65.40209466565399, 5.262639941411422, -3.0315939096971647], "high": [109.3323221736685, 56.29275767507379, -0.3934988005267144, -3.0772594822340906]}},
    "railroad_crossing": {"valid_min": 1.0, "valid_max": 40.0, "chart_type": "log", "coefficients": {"low": [0.5029257285845483, 5.749904110236876, 3.9821246564884927], "medium": [5.254000147362391, 11.422544085878982, 37.11519773645627, -16.910805393020667], "high": [19.340445737998966, 18.23862361469355, 53.47893872454435, -25.833753655092053]}},
    "raveling": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [1.3364599508023929, 2.5381741896122354, 2.2521180739800015], "medium": [8.547489838241672, 5.50999488268751, 2.8976928296955693, 1.6574499250743635], "high": [15.287212880590612, 12.713212203226831, 11.470876324575428, 5.038585870016526, -3.053732169374685]}},
    "rutting": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [7.166892989037484, 15.480212329682374, 7.204541988003889, -2.056356612335577], "medium": [17.07939650585985, 23.042120387468632, 7.267187817332259, -2.992255156045317], "high": [26.55168225206831, 25.25096737467123, 9.616695504985966, 2.709242740203141, -2.9515698599184823]}},
    "shoving": {"valid_min": 0.1, "valid_max": 50.0, "chart_type": "log", "coefficients": {"low": [4.413031420403843, 10.016571238629226, 5.535372489599174], "medium": [9.294794348392088, 16.13985140155734, 10.13135831372187], "high": [17.76171628163815, 19.396982653439373, 15.785981300450224, 3.016185582467493, -3.721051791249142]}},
    "slippage_cracking": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [4.248092940884805, 14.667313248550373, 9.293396804566436, -2.1419610585945215], "medium": [10.838179259718574, 20.247013730206078, 13.130140952344512, -0.24505223183555103, -2.039447457156605], "high": [17.75416989366112, 32.470845743485654, 29.87775942697452, -5.259819524132407, -13.039225691175872, 4.346079607761759]}},
    "swell": {"valid_min": 1.0, "valid_max": 30.0, "chart_type": "log", "coefficients": {"low": [0.2776553410390519, 12.183535319307815], "medium": [9.7262404794108, 26.04919313918952], "high": [33.021164499056795, 9.27090451459351, 10.656396250845475]}},
    "weathering": {"valid_min": 0.1, "valid_max": 100.0, "chart_type": "log", "coefficients": {"low": [1.3364599508023929, 2.5381741896122354, 2.2521180739800015], "medium": [8.547489838241672, 5.50999488268751, 2.8976928296955693, 1.6574499250743635], "high": [15.287212880590612, 12.713212203226831, 11.470876324575428, 5.038585870016526, -3.053732169374685]}}
}

CDV_ASPHALT_COEFFICIENTS = {
    "q1": [0.0, 0.9999999999999999],
    "q2": [-4.767827657379179, 0.9204018317853505, -0.001751870485036131],
    "q3": [-7.225000000000371, 0.8352547729618199, -0.0013491357069143565],
    "q4": [-12.07301341589303, 0.8359713622291058, -0.0013966718266254004],
    "q5": [-12.825902992776388, 0.7672961816305504, -0.0011578947368421164],
    "q6": [-15.061893704850672, 0.7418633900928823, -0.0010566950464396384],
    "q7": [-18.186068111455448, 0.8352115583075376, -0.0016336429308565648],
    "q8": [-11.661042311659124, 0.5804493464050489, 0.0012355521155837934, -9.988820089439906e-06],
    "q9": [-10.821138630889415, 0.5307870370368919, 0.0020743464052291917, -1.4092420593968467e-05],
    "q10": [-33.76740887897169, 2.045383029591749, -0.033305999495864574, 0.00035829181520672087, -1.7915566098874656e-06, 3.173417753551723e-09]
}

# ==========================================================
# FUNGSI PEMROSESAN SPASIAL & MATEMATIKA
# ==========================================================
def read_zip_shapefile(uploaded_file, tmpdir):
    """Membaca shapefile dari dalam file zip"""
    zip_path = os.path.join(tmpdir, uploaded_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    extract_dir = os.path.join(tmpdir, uploaded_file.name.replace('.zip', ''))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".shp"):
                return gpd.read_file(os.path.join(root, file))
    return None

def hitung_width(gdf):
    gdf = gdf.copy()
    width_list = []
    for geom in gdf.geometry:
        if geom.is_empty:
            width_list.append(np.nan)
            continue
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [LineString([coords[i], coords[i+1]]).length for i in range(4)]
        width_list.append(min(edges) * 1000)
    gdf["WIDTH_MM"] = width_list
    return gdf

def hitung_diameter_pothole(gdf):
    gdf = gdf.copy()
    diameter_list = []
    for geom in gdf.geometry:
        if geom.is_empty:
            diameter_list.append(np.nan)
            continue
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [LineString([coords[i], coords[i+1]]).length for i in range(4)]
        unique_edges = sorted(list(set([round(e,5) for e in edges])))
        if len(unique_edges) >= 2:
            diameter = (max(unique_edges) + min(unique_edges)) / 2
        else:
            diameter = unique_edges[0]
        diameter_list.append(diameter * 1000)
    gdf["DIAMETER_MM"] = diameter_list
    return gdf

def hitung_depth(gdf, dsm_path, buffer_distance=0.3):
    with rasterio.open(dsm_path) as DSM:
        dsm_crs = DSM.crs
        nodata_val = DSM.nodata

    if gdf.crs != dsm_crs:
        gdf = gdf.to_crs(dsm_crs)

    buffer_outer = gdf.geometry.buffer(buffer_distance)
    ring_geom = buffer_outer.difference(gdf.geometry)

    stats_hole = zonal_stats(gdf.geometry, dsm_path, stats=["percentile_10"], nodata=nodata_val)
    stats_ring = zonal_stats(ring_geom, dsm_path, stats=["median"], nodata=nodata_val)

    depth_list = []
    for i in range(len(gdf)):
        z_min = stats_hole[i]["percentile_10"]
        z_ref = stats_ring[i]["median"]
        depth = (z_ref - z_min) * 1000 if (z_min is not None and z_ref is not None) else 0
        depth = max(0, min(depth, 80)) # Clamp realistis
        depth_list.append(depth)

    gdf = gdf.copy()
    gdf["DEPTH_MM"] = depth_list
    return gdf

def tentukan_severity(distress_type, row):
    distress = distress_type.lower()
    depth = row.get("Depth_mm", 0)
    width = row.get("Width_mm", 0)
    diameter = row.get("Diameter_mm", 0)
    area = row.geometry.area if row.geometry is not None else 0

    if "alligator" in distress: return "Low" if width < 10 else "Medium" if width <= 25 else "High"
    if "bleeding" in distress: return "Low" if area < 2 else "Medium" if area < 6 else "High"
    if "block" in distress: return "Low" if width < 10 else "Medium" if width <= 25 else "High"
    if "bump" in distress or "sag" in distress: return "Low" if depth <= 10 else "Medium" if depth <= 25 else "High"
    if "corrugation" in distress: return "Low" if depth <= 10 else "Medium" if depth <= 25 else "High"
    if "depression" in distress: return "Low" if depth <= 25 else "Medium" if depth <= 50 else "High"
    if "edge" in distress: return "Low" if width < 10 else "Medium" if width <= 25 else "High"
    if "joint" in distress or "reflection" in distress: return "Low" if width < 10 else "Medium" if width <= 75 else "High"
    if "shoulder" in distress: return "Low" if depth <= 50 else "Medium" if depth <= 100 else "High"
    if "longitudinal" in distress or "transverse" in distress: return "Low" if width < 10 else "Medium" if width <= 75 else "High"
    if "patch" in distress: return "Low" if area < 1 else "Medium" if area < 3 else "High"
    if "polished" in distress: return "Low" if area < 5 else "Medium" if area < 15 else "High"
    if "pothole" in distress:
        if depth <= 25: return "Low" if diameter < 450 else "Medium"
        elif depth <= 50: return "Low" if diameter < 200 else "Medium" if diameter < 450 else "High"
        else: return "Medium" if diameter < 450 else "High"
    if "railroad" in distress: return "Low" if depth <= 25 else "Medium" if depth <= 75 else "High"
    if "ravel" in distress: return "Low" if area < 5 else "Medium" if area < 20 else "High"
    if "rutting" in distress: return "Low" if depth <= 13 else "Medium" if depth <= 25 else "High"
    if "shoving" in distress: return "Low" if depth <= 10 else "Medium" if depth <= 25 else "High"
    if "slippage" in distress: return "Low" if width < 10 else "Medium" if width <= 40 else "High"
    if "swell" in distress: return "Low" if depth <= 25 else "Medium" if depth <= 75 else "High"
    if "weather" in distress: return "Low" if area < 5 else "Medium" if area < 20 else "High"
    return "Low"

def evaluate_polynomial(coeffs, x):
    return sum(c * (x ** i) for i, c in enumerate(coeffs))

def lookup_dv(distress_type, severity, density):
    key = distress_type.lower().replace(" ", "_").replace("-", "_")
    if key not in DISTRESS_COEFFICIENTS: return 0.0
    data = DISTRESS_COEFFICIENTS[key]
    coeffs = data["coefficients"].get(severity.lower())
    if not coeffs: return 0.0
    
    density = max(data["valid_min"], min(data["valid_max"], density))
    val_to_eval = math.log10(density) if data["chart_type"] == "log" else density
    dv = evaluate_polynomial(coeffs, val_to_eval)
    return max(0.0, min(100.0, dv))

def lookup_cdv_asphalt(q, total_deduct_value):
    q_lookup = f"q{min(max(int(q), 1), 10)}"
    coeffs = CDV_ASPHALT_COEFFICIENTS.get(q_lookup)
    tdv_clamped = max(0.0, min(200.0, total_deduct_value))
    cdv = evaluate_polynomial(coeffs, tdv_clamped)
    return max(0.0, min(100.0, cdv))

def rating_pci(pci):
    if pci > 85: return "Good"
    elif pci > 70: return "Satisfactory"
    elif pci > 55: return "Fair"
    elif pci > 40: return "Poor"
    elif pci > 25: return "Very Poor"
    elif pci > 10: return "Serious"
    else: return "Failed"

# =========================================
# TAMPILAN SIDEBAR
# =========================================
with st.sidebar:
    st.header("📝 Informasi Survey")
    lokasi = st.text_input("Lokasi Survey", "Jl. Jawa Raya")
    sta_umum = st.text_input("STA Umum", "0+000 - 1+000")
    surveyor = st.text_input("Nama Surveyor", "Nama Anda")
    tanggal = st.text_input("Tanggal Survey", "27 Februari 2026")
    instansi = st.text_input("Instansi", "Universitas Diponegoro")
    
    st.header("⚙️ Parameter Jalan")
    lebar_jalan = st.number_input("Lebar Jalan (m)", value=3.5, step=0.1)
    interval_segmen = st.number_input("Interval Segmen (m)", value=100, step=10)
    epsg_code = st.number_input("Kode EPSG UTM Lokal (Contoh: 32749 untuk Jawa Tengah)", value=32749, step=1)

# =========================================
# TAMPILAN UTAMA (UPLOAD FILES)
# =========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 1. Data Dasar")
    jalan_file = st.file_uploader("Upload Shapefile Jalan (.zip)", type="zip")
    
    dsm_mode = st.radio("Cara Input Data DSM:", ["Upload File .tif", "Paste Link Google Drive"])
    dsm_file = None
    dsm_link = ""
    
    if dsm_mode == "Upload File .tif":
        dsm_file = st.file_uploader("Upload Data DSM (.tif)", type="tif")
    else:
        dsm_link = st.text_input("Paste Link Shareable Google Drive (.tif)")
        st.caption("Pastikan akses link Google Drive diatur ke 'Anyone with the link' (Siapa saja yang memiliki link).")

with col2:
    st.subheader("⚠️ 2. Data Kerusakan (Distress)")
    distress_keys = list(DISTRESS_COEFFICIENTS.keys())
    distress_options = [k.replace('_', ' ').title() for k in distress_keys]
    selected_distress = st.multiselect("Pilih Kerusakan yang Ditemukan:", distress_options)
    
    uploaded_distress = {}
    for d in selected_distress:
        file = st.file_uploader(f"Upload SHP {d} (.zip)", type="zip", key=d)
        if file:
            original_key = d.lower().replace(' ', '_')
            uploaded_distress[original_key] = file

st.divider()

# =========================================
# PROSES UTAMA (EKSEKUSI)
# =========================================
if st.button("🚀 Proses & Hitung PCI", type="primary", use_container_width=True):
    
    is_dsm_valid = False
    if dsm_mode == "Upload File .tif" and dsm_file is not None:
        is_dsm_valid = True
    elif dsm_mode == "Paste Link Google Drive" and dsm_link != "":
        is_dsm_valid = True

    if not jalan_file or not uploaded_distress or not is_dsm_valid:
        st.error("⚠️ Mohon lengkapi Shapefile Jalan, Data DSM (Upload/Link), dan minimal 1 Data Kerusakan.")
    else:
        with st.spinner("Memproses Analisis Geospasial... (Ini mungkin memakan waktu beberapa menit)"):
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    # 1. BACA JALAN & BUAT SEGMEN
                    jalan = read_zip_shapefile(jalan_file, tmpdir)
                    if jalan.crs is None:
                        st.error("CRS shapefile jalan tidak terdefinisi!")
                        st.stop()
                    if jalan.crs.is_geographic:
                        jalan = jalan.to_crs(epsg=epsg_code)
                    
                    union_geom = jalan.geometry.union_all()
                    merged_line = linemerge(union_geom) if union_geom.geom_type == "MultiLineString" else union_geom
                    
                    panjang_total = merged_line.length
                    segments = [LineString([merged_line.interpolate(start), merged_line.interpolate(min(start + interval_segmen, panjang_total))]) 
                                for start in np.arange(0, panjang_total, interval_segmen)]
                    
                    seg_gdf = gpd.GeoDataFrame(geometry=segments, crs=jalan.crs)
                    seg_gdf["Segmen"] = range(1, len(seg_gdf)+1)
                    seg_gdf["geometry"] = seg_gdf.buffer(lebar_jalan / 2, cap_style=2)
                    seg_gdf["Unit_Area"] = seg_gdf.geometry.area
                    
                    # 2. SIMPAN ATAU DOWNLOAD DSM
                    dsm_path = os.path.join(tmpdir, "dsm.tif")
                    
                    if dsm_mode == "Upload File .tif":
                        with open(dsm_path, "wb") as f:
                            f.write(dsm_file.getbuffer())
                            
                    elif dsm_mode == "Paste Link Google Drive":
                        st.info("⏳ Mengunduh DSM dari Google Drive... (Ini sangat cepat!)")
                        import gdown
                        import re
                        match = re.search(r"/d/([a-zA-Z0-9_-]+)", dsm_link)
                        if match:
                            file_id = match.group(1)
                            gdown.download(id=file_id, output=dsm_path, quiet=False)
                        else:
                            st.error("❌ Link Google Drive tidak valid. Pastikan format link benar.")
                            st.stop()

                    # 3. BACA & PROSES DISTRESS
                    distress_layers = {}
                    for key, file in uploaded_distress.items():
                        gdf = read_zip_shapefile(file, tmpdir)
                        if gdf is not None:
                            distress_layers[key] = gdf

                    all_distress_list = []
                    target_crs = seg_gdf.crs # Ambil patokan absolut
                    
                    for nama_distress, gdf in distress_layers.items():
                        if gdf.empty: continue
                        
                        gdf = gdf.copy()
                        if gdf.crs is None:
                            gdf.set_crs(target_crs, inplace=True, allow_override=True)
                        elif gdf.crs != target_crs:
                            gdf = gdf.to_crs(target_crs)
                        
                        if any(x in nama_distress for x in ["crack", "alligator", "block", "long", "slip", "joint", "edge"]):
                            gdf = hitung_width(gdf)
                        if any(x in nama_distress for x in ["rutting", "depression", "corrugation", "bump", "sag", "shoving", "swell", "shoulder", "railroad"]):
                            gdf = hitung_depth(gdf, dsm_path)
                        if "pothole" in nama_distress:
                            gdf = hitung_depth(gdf, dsm_path)
                            gdf = hitung_diameter_pothole(gdf)
                            
                        gdf["Distress_Type"] = nama_distress
                        gdf["Severity"] = gdf.apply(lambda row: tentukan_severity(nama_distress, row), axis=1)
                        gdf["Priority"] = gdf["Severity"].map({"High": 3, "Medium": 2, "Low": 1}).fillna(1)
                        
                        all_distress_list.append(gdf)

                    # 4. OVERLAY & FLATTEN
                    df_detail_list = []
                    
                    if all_distress_list:
                        cleaned_layers = []
                        for gdf in all_distress_list:
                            if gdf.crs != seg_gdf.crs:
                                gdf = gdf.to_crs(seg_gdf.crs)
                            gdf = gdf.set_crs(seg_gdf.crs, allow_override=True)
                            cleaned_layers.append(gdf)
                    
                        master_distress = gpd.GeoDataFrame(pd.concat(cleaned_layers, ignore_index=True), crs=seg_gdf.crs)
                        master_distress = master_distress.sort_values(by="Priority", ascending=False).reset_index(drop=True)
                    
                        accumulated_geom = Polygon()
                        cleaned_geometries = []
                    
                        for idx, row in master_distress.iterrows():
                            geom = row.geometry
                            new_geom = geom.difference(accumulated_geom) if not geom.is_empty else geom
                            cleaned_geometries.append(new_geom)
                            if not new_geom.is_empty:
                                accumulated_geom = accumulated_geom.union(new_geom)
                    
                        master_distress["geometry"] = cleaned_geometries
                        master_distress = master_distress[~master_distress.geometry.is_empty]
                        inter_all = gpd.overlay(master_distress, seg_gdf, how="intersection")
                    
                        if not inter_all.empty:
                            inter_all["Area_Intersect"] = inter_all.geometry.area
                            agg_df = inter_all.groupby(['Segmen', 'Distress_Type', 'Severity', 'Unit_Area'])['Area_Intersect'].sum().reset_index()
                    
                            for _, row in agg_df.iterrows():
                                density = max(0, min(100, (row['Area_Intersect'] / row['Unit_Area']) * 100))
                                dv = lookup_dv(row['Distress_Type'], row['Severity'], density)
                                df_detail_list.append({
                                    "Segmen": row["Segmen"],
                                    "Distress": row['Distress_Type'],
                                    "Severity": row["Severity"],
                                    "Density": density,
                                    "DV": dv
                                })

                    df_detail = pd.DataFrame(df_detail_list)
                    
                    # 5. HITUNG PCI
                    df_pci_list = []
                    for seg_id in seg_gdf["Segmen"]:
                        df_seg = df_detail[df_detail["Segmen"] == seg_id] if not df_detail.empty else pd.DataFrame()
                        if df_seg.empty:
                            df_pci_list.append({"Segmen": seg_id, "TDV": 0, "q": 0, "CDV": 0, "PCI": 100})
                            continue
                        
                        dvs = np.array(sorted(df_seg["DV"].tolist(), reverse=True))
                        hdv = dvs[0]
                        m = min(10.0, 1 + (9.0 / 95.0) * (100.0 - hdv))
                        m_int = int(np.floor(m))
                        
                        entered_dvs = dvs[:m_int + 1].copy() if len(dvs) > m_int else np.copy(dvs)
                        if len(dvs) > m_int: entered_dvs[-1] *= (m - m_int)
                        
                        max_cdv = 0.0
                        while True:
                            q = np.sum(entered_dvs > 2.0)
                            if q == 0 and max_cdv != 0: break
                            cdv_current = lookup_cdv_asphalt(q, np.sum(entered_dvs))
                            if cdv_current > max_cdv: max_cdv = cdv_current
                            if q <= 1: break
                            
                            gt_2_indices = np.where(entered_dvs > 2.0)[0]
                            if len(gt_2_indices) > 0:
                                entered_dvs[gt_2_indices[np.argmin(entered_dvs[gt_2_indices])]] = 2.0
                            else: break
                            
                        df_pci_list.append({
                            "Segmen": seg_id, "TDV": round(np.sum(entered_dvs), 2), "q": q,
                            "CDV": round(max_cdv, 2), "PCI": round(max(0, min(100, 100 - max_cdv)), 2)
                        })
                    
                    df_pci = pd.DataFrame(df_pci_list)
                    df_pci["Rating"] = df_pci["PCI"].apply(rating_pci)
                    df_pci["STA"] = df_pci["Segmen"].apply(lambda x: f"{(x-1)*interval_segmen} - {x*interval_segmen} m")
                    
                    seg_gdf = seg_gdf.merge(df_pci, on="Segmen", how="left")
                    seg_gdf["PCI"] = seg_gdf["PCI"].fillna(100)
                    seg_gdf["Rating"] = seg_gdf["Rating"].fillna("Good")

                    # =========================================
                    # VISUALISASI PETA & GRAFIK
                    # =========================================
                    warna_pci = {"Good": "#006400", "Satisfactory": "#8FBC8F", "Fair": "#FFFF00", "Poor": "#FF6347", "Very Poor": "#FF4500", "Serious": "#8B0000", "Failed": "#A9A9A9"}
                    
                    # Peta
                    fig_map, ax_map = plt.subplots(figsize=(10,6))
                    seg_plot = seg_gdf.copy()
                    seg_plot["geometry"] = seg_plot.geometry.buffer(4)
                    
                    legend_handles = []
                    
                    for rating, warna in warna_pci.items():
                        subset = seg_plot[seg_plot["Rating"] == rating]
                        if not subset.empty:
                            subset.plot(ax=ax_map, color=warna, edgecolor="black", label=f"{rating}")
                            legend_handles.append(mpatches.Patch(color=warna, label=f"{rating} ({len(subset)})"))
                    
                    for idx, row in seg_gdf.iterrows():
                        centroid = row.geometry.centroid
                        ax_map.text(
                            centroid.x, centroid.y,
                            f"S{row['Segmen']}\n{row['PCI']:.0f}",
                            fontsize=7, weight="bold", ha="center", va="center",
                            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.2", edgecolor="gray", lw=0.5)
                        )
                    
                    if legend_handles:
                        ax_map.legend(handles=legend_handles, loc="best", title="Kategori PCI", fontsize=8, title_fontsize=9)
                    
                    ax_map.axis("off")
                    peta_path = os.path.join(tmpdir, "peta_pci.png")
                    plt.savefig(peta_path, dpi=300, bbox_inches='tight')
                    plt.close(fig_map)
                    
                    # Grafik
                    fig_bar, ax_bar = plt.subplots(figsize=(6,4))
                    rekap = seg_gdf["Rating"].value_counts()
                    warna_bar = [warna_pci.get(x, "grey") for x in rekap.index]
                    rekap.plot(kind="bar", color=warna_bar, edgecolor="black", ax=ax_bar)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    grafik_path = os.path.join(tmpdir, "grafik_pci.png")
                    plt.savefig(grafik_path, dpi=300)
                    plt.close(fig_bar)
                    
                    # =========================================
                    # PEMBUATAN PDF
                    # =========================================
                    pdf_path = os.path.join(tmpdir, "Laporan_PCI.pdf")
                    doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.landscape(pagesizes.A4))
                    elements = []
                    styles = getSampleStyleSheet()
                    
                    elements.append(Paragraph(f"<font size=18><b>{instansi}</b></font>", ParagraphStyle('Title', alignment=TA_CENTER)))
                    elements.append(Spacer(1, 0.5*inch))
                    elements.append(Paragraph("<font size=24>LAPORAN SURVEY PAVEMENT CONDITION INDEX (PCI)</font>", ParagraphStyle('Title', alignment=TA_CENTER)))
                    elements.append(Spacer(1, 0.5*inch))
                    elements.append(Paragraph(f"<b>Lokasi :</b> {lokasi} | <b>STA :</b> {sta_umum}", styles["Normal"]))
                    elements.append(Paragraph(f"<b>Surveyor :</b> {surveyor} | <b>Tanggal :</b> {tanggal}", styles["Normal"]))
                    elements.append(PageBreak())
                    
                    elements.append(Paragraph("<b>HASIL ANALISIS KONDISI JALAN</b>", styles["Heading2"]))
                    elements.append(Image(peta_path, width=7.5*inch, height=4.5*inch))
                    elements.append(PageBreak())
                    
                    for idx, seg in df_pci.sort_values('Segmen').reset_index(drop=True).iterrows():
                        if idx > 0: elements.append(PageBreak())
                        seg_id = seg["Segmen"]
                        df_seg_detail = df_detail[df_detail["Segmen"] == seg_id] if not df_detail.empty else pd.DataFrame()
                        
                        hdv_val = df_seg_detail["DV"].max() if not df_seg_detail.empty else 0.0
                        m_val = min(1 + (9.0 / 95.0) * (100.0 - hdv_val), 10.0) if hdv_val > 0 else 0.0
                        
                        elements.append(Paragraph(f"<b>REPORT SEGMEN : {seg_id} (STA: {seg['STA']})</b>", styles["Heading3"]))
                        elements.append(Spacer(1, 0.1*inch))
                        elements.append(Paragraph("<b>A. Flexible Pavement Condition Data Sheet</b>", styles["Normal"]))
                        
                        tabel_d_data = [["Distress Type", "Severity", "Quantity (sq.m)", "", "", "", "", "", "", "", "", "", "Total", "Density %", "DV"],
                                        ["", "", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "", "", ""]]
                        if df_seg_detail.empty:
                            tabel_d_data.append(["Tidak ada kerusakan", "-", "", "", "", "", "", "", "", "", "", "", "0.00", "0.00", "0.00"])
                        else:
                            for _, row in df_seg_detail.iterrows():
                                clean_name = row["Distress"].replace("_", " ").title()
                                tqty = (row["Density"] / 100.0) * (interval_segmen * lebar_jalan)
                                tabel_d_data.append([clean_name, row["Severity"], f"{tqty:.2f}", "", "", "", "", "", "", "", "", "", f"{tqty:.2f}", f"{row['Density']:.2f}", f"{row['DV']:.2f}"])
                        
                        col_w_d = [2.2*inch, 0.7*inch] + [0.35*inch]*10 + [0.6*inch, 0.7*inch, 0.6*inch]
                        t_d = Table(tabel_d_data, colWidths=col_w_d, repeatRows=2)
                        t_d.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,1), colors.lightgrey), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8),
                                                 ('SPAN', (0,0), (0,1)), ('SPAN', (1,0), (1,1)), ('SPAN', (2,0), (11,0)), ('SPAN', (12,0), (12,1)), ('SPAN', (13,0), (13,1)), ('SPAN', (14,0), (14,1))]))
                        elements.append(t_d)
                        elements.append(Spacer(1, 0.2*inch))
                        
                        bg_color = warna_pci.get(seg['Rating'], "#FFFFFF")
                        txt_color = colors.black if seg['Rating'] in ["Satisfactory", "Fair", "Good"] else colors.white
                        t_sub = Table(
                            [["HDV", "m (Max Allowable)", "Max CDV", "PCI", "Rating (ASTM)"],
                             [f"{hdv_val:.2f}", f"{m_val:.2f}", f"{seg['CDV']:.2f}", f"{seg['PCI']:.2f}", seg['Rating']]],
                            colWidths=[1.6*inch, 1.6*inch, 1.6*inch, 1.8*inch, 2.2*inch]
                        )
                        
                        t_sub.setStyle(TableStyle([
                            ('GRID', (0,0), (-1,-1), 0.6, colors.grey),
                            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                            ('FONTSIZE', (0,0), (-1,0), 10),
                            ('FONTSIZE', (0,1), (-1,1), 12),
                            ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
                            # Highlight PCI cell (kolom ke-4 baris ke-2)
                            ('BACKGROUND', (3,1), (3,1), colors.HexColor(bg_color)),
                            ('TEXTCOLOR', (3,1), (3,1), txt_color),
                            ('ALIGN', (3,1), (3,1), 'CENTER'),
                            # Bold rating cell and give border
                            ('BACKGROUND', (4,1), (4,1), colors.whitesmoke),
                            ('BOX', (4,1), (4,1), 1.0, colors.HexColor(bg_color)),
                            ('FONTSIZE', (4,1), (4,1), 11),
                            ('FONTNAME', (4,1), (4,1), 'Helvetica-Bold')
                        ]))
                        elements.append(t_sub)
                        
                    doc.build(elements)
                    
                    # =========================================
                    # SIMPAN KE SESSION STATE SEBELUM TEMP DIHAPUS
                    # =========================================
                    st.session_state.df_pci = df_pci
                    st.session_state.df_detail = df_detail
                    
                    with open(peta_path, "rb") as f:
                        st.session_state.peta_bytes = f.read()
                    with open(grafik_path, "rb") as f:
                        st.session_state.grafik_bytes = f.read()
                    with open(pdf_path, "rb") as f:
                        st.session_state.pdf_bytes = f.read()
                        
                    # Beri tanda bahwa proses sudah 100% komplit
                    st.session_state.proses_selesai = True

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
                    st.session_state.proses_selesai = False

# =========================================
# TAMPILKAN HASIL DI WEB (DARI SESSION STATE)
# =========================================
# Bagian ini di luar blok tombol "Proses" agar tidak hilang saat dropdown diklik
if st.session_state.proses_selesai:
    st.success("✅ Analisis Berhasil!")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("Peta Kondisi PCI")
        st.image(st.session_state.peta_bytes)
    with col_res2:
        st.subheader("Distribusi")
        st.image(st.session_state.grafik_bytes)
        st.metric("Rata-rata PCI", round(st.session_state.df_pci["PCI"].mean(), 2))
    
    st.markdown("---")
    
    # -----------------------------------------------------
    # FITUR BARU: SKALA RATING & TABEL REKAPITULASI (DIPERBAIKI)
    # -----------------------------------------------------
    col_tab, col_leg = st.columns([2, 1])
    
    with col_tab:
        st.subheader("Tabel Rekapitulasi Umum")
        # Menyembunyikan Index Tabel (0, 1, 2, 3...)
        st.dataframe(
            st.session_state.df_pci[["Segmen", "STA", "TDV", "CDV", "PCI", "Rating"]], 
            use_container_width=True, 
            hide_index=True
        )

    with col_leg:
        # Menggunakan format st.markdown yang lebih aman dan rapi
        st.markdown("<h4 style='text-align: center; margin-bottom: 15px;'>Skala Rating PCI</h4>", unsafe_allow_html=True)
        
        skala_pci = [
            ("Good", "#006400", "white", "85 - 100"),
            ("Satisfactory", "#8FBC8F", "black", "70 - 85"),
            ("Fair", "#FFFF00", "black", "55 - 70"),
            ("Poor", "#FF6347", "white", "40 - 55"),
            ("Very Poor", "#FF4500", "white", "25 - 40"),
            ("Serious", "#8B0000", "white", "10 - 25"),
            ("Failed", "#A9A9A9", "black", "0 - 10")
        ]
        
        for nama, bg, txt, rentang in skala_pci:
            # HTML dipecah per baris agar tidak error saat dibaca Streamlit
            html_baris = f"""
            <div style='background-color: {bg}; color: {txt}; padding: 10px; margin-bottom: 5px; border-radius: 5px; display: flex; justify-content: space-between; font-weight: bold;'>
                <span>{nama}</span>
                <span>{rentang}</span>
            </div>
            """
            st.markdown(html_baris, unsafe_allow_html=True)

    # =========================================
    # FITUR DASHBOARD DETAIL PER SEGMEN
    # =========================================
    st.markdown("---")
    st.subheader("🔎 Dashboard Detail Per Segmen")
    st.markdown("Pilih nomor segmen di bawah ini untuk melihat rincian perhitungan setara lembar kerja ASTM.")

    # Dropdown memilih segmen
    list_segmen = st.session_state.df_pci["Segmen"].tolist()
    pilihan_segmen = st.selectbox("Pilih Segmen:", list_segmen)

    if pilihan_segmen:
        # Ambil memori
        df_pci_mem = st.session_state.df_pci
        df_det_mem = st.session_state.df_detail

        # Ambil data spesifik
        seg_data = df_pci_mem[df_pci_mem["Segmen"] == pilihan_segmen].iloc[0]
        df_seg_detail = df_det_mem[df_det_mem["Segmen"] == pilihan_segmen]

        # Hitung HDV dan m
        hdv_val = df_seg_detail["DV"].max() if not df_seg_detail.empty else 0.0
        m_val = min(1 + (9.0 / 95.0) * (100.0 - hdv_val), 10.0) if hdv_val > 0 else 0.0

        st.markdown(f"#### REPORT SEGMEN : {pilihan_segmen} (STA: {seg_data['STA']})")

        # TABEL A
        st.markdown("**A. Flexible Pavement Condition Data Sheet**")
        if df_seg_detail.empty:
            st.info("✅ Tidak ada kerusakan pada segmen ini.")
        else:
            display_df = df_seg_detail.copy()
            display_df["Distress Type"] = display_df["Distress"].str.replace("_", " ").str.title()
            display_df["Quantity (sq.m)"] = (display_df["Density"] / 100.0) * (interval_segmen * lebar_jalan)
            display_df = display_df[["Distress Type", "Severity", "Quantity (sq.m)", "Density", "DV"]]
            display_df.rename(columns={"Density": "Density (%)", "DV": "Deduct Value (DV)"}, inplace=True)
            
            # Menggunakan st.dataframe dengan hide_index=True agar nomor index hilang
            st.dataframe(
                display_df.style.format({"Quantity (sq.m)": "{:.2f}", "Density (%)": "{:.2f}", "Deduct Value (DV)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True
            )

        # -----------------------------
        # GANTI: TABEL B (Maximum allowable number of distresses (m))
        # -----------------------------
        st.markdown("**B. Maximum allowable number of distresses (m)**")
        col_b1, col_b2 = st.columns([1,1], gap="large")
        with col_b1:
            st.markdown(f"""
            <div style="background:#f7fbff;border:1px solid #d0e6ff;padding:12px;border-radius:10px;text-align:center;">
              <div style="font-size:13px;color:#34495e;font-weight:600;">Highest Deduct Value (HDV)</div>
              <div style="font-size:32px;font-weight:800;margin-top:6px;color:#0b5394;">{hdv_val:.2f}</div>
              <div style="font-size:11px;color:#566573;margin-top:6px;">(DV tertinggi yang ditemukan pada segmen)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b2:
            st.markdown(f"""
            <div style="background:#fffaf0;border:1px solid #ffe6b3;padding:12px;border-radius:10px;text-align:center;">
              <div style="font-size:13px;color:#4d3b00;font-weight:600;">m = 1 + (9/95)*(100 - HDV)  (maks 10)</div>
              <div style="font-size:32px;font-weight:800;margin-top:6px;color:#b35900;">{m_val:.2f}</div>
              <div style="font-size:11px;color:#7a6b47;margin-top:6px;">(Jumlah maksimum distress yang dipertimbangkan)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption("HDV dan m digunakan untuk menentukan berapa banyak Deduct Values yang masuk perhitungan CDV.")

    # -----------------------------
    # GANTI: TABEL C (Calculate Pavement Condition Index)
    # -----------------------------
    st.markdown("**C. Calculate Pavement Condition Index (PCI)**")
    col_c1, col_c2, col_c3 = st.columns([1,1,1], gap="large")
    
    # Max CDV
    with col_c1:
        st.markdown(f"""
        <div style="background:#f3f8f1;border:1px solid #d7efd8;padding:12px;border-radius:10px;text-align:center;">
          <div style="font-size:13px;color:#1b5e20;font-weight:600;">Max CDV</div>
          <div style="font-size:28px;font-weight:800;margin-top:6px;color:#145214;">{seg_data['CDV']:.2f}</div>
          <div style="font-size:11px;color:#4b6b4b;margin-top:6px;">(Total deduct value yang menentukan PCI)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # PCI + progress bar
    with col_c2:
        pci_val = seg_data['PCI']
        st.markdown(f"""
        <div style="padding:6px;border-radius:10px;text-align:center;">
          <div style="font-size:13px;color:#222;font-weight:600;">PCI = 100 - Max_CDV</div>
          <div style="font-size:36px;font-weight:900;margin-top:6px;">{pci_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        # progress bar visual (normalisasi ke 0..1)
        st.progress(min(max(pci_val/100.0, 0.0), 1.0))
    
    # Rating card (besar & kontras)
    with col_c3:
        bg_col = warna_pci_dict.get(seg_data['Rating'], "#FFFFFF")
        txt_col = "black" if seg_data['Rating'] in ["Satisfactory", "Fair", "Good"] else "white"
        st.markdown(f"""
        <div style="background:{bg_col}; color:{txt_col}; padding:18px; border-radius:10px; text-align:center; border:1px solid #ccc;">
          <div style="font-size:12px;font-weight:600;">Rating (ASTM)</div>
          <div style="font-size:28px;font-weight:900;margin-top:8px;">{seg_data['Rating']}</div>
          <div style="font-size:11px;margin-top:6px;">(Kategori kondisi perkerasan menurut ASTM D6433)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption("PCI di atas juga divisualkan sebagai progress bar untuk memudahkan pembacaan kondisi secara cepat.")
    
    # Tombol Download PDF
    st.download_button(
        label="📄 Download Laporan Full PDF (ASTM Data Sheet)",
        data=st.session_state.pdf_bytes,
        file_name=f"PCI_{lokasi.replace(' ', '_')}.pdf",
        mime="application/pdf",
        type="primary"
    )

