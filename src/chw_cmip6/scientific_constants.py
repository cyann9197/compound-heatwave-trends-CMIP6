"""Manuscript-level scientific contracts shared by analysis and tests."""

ANALYSIS_PERIOD = (1981, 2014)
ANALYSIS_MONTHS = (6, 7, 8)
N_MODELS = 30

# 区域边界顺序为 lon_min, lon_max, lat_min, lat_max。
GIC_BOUNDS = (-55.0, -22.0, 71.0, 83.0)

FIG3_VARIABLES = ("tcc", "net", "q2m", "eddy_z500")
EXTENDED_DATA_4_7_VARIABLES = ("net_s", "net_l", "rlds", "pr", "e", "ef")
EXTENDED_DATA_5_6_VARIABLES = (
    "tmax",
    "tmin",
    "q2m",
    "e",
    "ef",
    "pr",
    "tcc",
    "net",
    "net_s",
    "net_l",
    "rlds",
    "eddy_z500",
)
FIG6_PRODUCT_TAG = "chwcumheatcfexcess_chwcalendar31d"
EXTENDED_DATA_8_PRODUCT_TAG = "linearbudget_chwcalendar31d"

OVER_ESTIMATED_REGIONS = ("NNA", "NAF", "EEU", "SAS", "ESB")
UNDER_ESTIMATED_REGIONS = ("SNA", "GIC", "WEU", "ENA")

MODEL_NAMES = (
    "ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-ESM1", "CanESM5",
    "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "E3SM-2-0", "E3SM-2-0-NARRM",
    "EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-CC", "EC-Earth3-Veg-LR", "FGOALS-f3-L",
    "FGOALS-g3", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "IITM-ESM", "INM-CM4-8",
    "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC6", "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL",
)
