"""Edit this file only if your shapefile paths / attribute names change."""

# ---- Inbuilt boundaries -------------------------------------------------
DISTRICT_SHP = r"D:/drive F/ASHISH/App/Admin_boundary/district_boundary_utm43.shp"
TALUKA_SHP   = r"D:/drive F/ASHISH\App/Admin_boundary/1Tahsil_boundary356.shp"

# Attribute column holding the names. None = auto-detect
# (looks for columns containing 'district' / 'tahsil' / 'taluka' / 'tehsil' / 'name').
DISTRICT_NAME_FIELD = "DTENAME"      # e.g. "DISTRICT"
TALUKA_NAME_FIELD   = "THENAME"      # e.g. "TAHSIL"

# Used ONLY if a shapefile has no .prj file
DISTRICT_FALLBACK_EPSG = 32643  # UTM 43N
TALUKA_FALLBACK_EPSG   = 4326

# ---- Defaults shown in the GUI -----------------------------------------
OTHER_CLASS_NAME = "Other crop"
