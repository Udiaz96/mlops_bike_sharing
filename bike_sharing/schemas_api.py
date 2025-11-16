from pydantic import BaseModel
from datetime import date


### Modelo Raw
class BikeSharingRaw(BaseModel):
    instant: int
    dteday: date
    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float
    casual: int
    registered: int
    cnt: int

class PredictionRequestRaw(BaseModel):
    """Esquema para la solicitud de predicción."""
    model_name: str # Por ejemplo: ElasticNet_Casual_BikeSharing
    features: BikeSharingRaw


# --- Definición del Esquema de Datos con 177 Características ---

# NOTA CRÍTICA: El orden de los campos aquí DEBE coincidir exactamente con el orden
# de las columnas que tu modelo recibió durante el entrenamiento.

class BikeSharingFeatures(BaseModel):
    """Esquema completo de las 177 características de entrada para la predicción."""
    # 4 Características Log-Transformadas (NO LOG-OBJETIVO)
    temp_log: float
    atemp_log: float
    hum_log: float
    windspeed_log: float
    
    # 6 Características Cíclicas (Fecha/Hora)
    mnth_sin: float
    mnth_cos: float
    hr_sin: float
    hr_cos: float
    weekday_sin: float
    weekday_cos: float
    
    # 4 Características de Estación y Año
    season_2: int
    season_3: int
    season_4: int
    yr_1: int
    
    # 161 Características One-Hot Encoded (weathersit_XX, holiday_1, workingday_1)
    # Estas deben estar en el orden exacto de tu procesamiento
    weathersit_2: int
    weathersit_3: int
    weathersit_4: int
    weathersit_11: int
    weathersit_14: int
    weathersit_15: int
    weathersit_19: int
    weathersit_21: int
    weathersit_22: int
    weathersit_24: int
    weathersit_26: int
    weathersit_31: int
    weathersit_32: int
    weathersit_34: int
    weathersit_35: int
    weathersit_37: int
    weathersit_38: int
    weathersit_40: int
    weathersit_44: int
    weathersit_46: int
    weathersit_50: int
    weathersit_51: int
    weathersit_52: int
    weathersit_53: int
    weathersit_54: int
    weathersit_57: int
    weathersit_61: int
    weathersit_62: int
    weathersit_63: int
    weathersit_64: int
    weathersit_67: int
    weathersit_68: int
    weathersit_69: int
    weathersit_70: int
    weathersit_76: int
    weathersit_79: int
    weathersit_80: int
    weathersit_81: int
    weathersit_83: int
    weathersit_86: int
    weathersit_87: int
    weathersit_90: int
    weathersit_92: int
    weathersit_93: int
    weathersit_94: int
    weathersit_95: int
    weathersit_97: int
    weathersit_98: int
    weathersit_99: int
    weathersit_103: int
    weathersit_106: int
    weathersit_108: int
    weathersit_112: int
    weathersit_113: int
    weathersit_124: int
    weathersit_126: int
    weathersit_129: int
    weathersit_139: int
    weathersit_142: int
    weathersit_144: int
    weathersit_148: int
    weathersit_150: int
    weathersit_152: int
    weathersit_154: int
    weathersit_155: int
    weathersit_156: int
    weathersit_162: int
    weathersit_176: int
    weathersit_182: int
    weathersit_185: int
    weathersit_190: int
    weathersit_192: int
    weathersit_197: int
    weathersit_199: int
    weathersit_204: int
    weathersit_205: int
    weathersit_207: int
    weathersit_231: int
    weathersit_247: int
    weathersit_254: int
    weathersit_264: int
    weathersit_280: int
    weathersit_289: int
    weathersit_295: int
    weathersit_299: int
    weathersit_306: int
    weathersit_310: int
    weathersit_311: int
    weathersit_313: int
    weathersit_344: int
    weathersit_345: int
    weathersit_352: int
    weathersit_354: int
    weathersit_371: int
    weathersit_389: int
    weathersit_394: int
    weathersit_412: int
    weathersit_422: int
    weathersit_432: int
    weathersit_433: int
    weathersit_435: int
    weathersit_453: int
    weathersit_454: int
    weathersit_466: int
    weathersit_472: int
    weathersit_476: int
    weathersit_480: int
    weathersit_483: int
    weathersit_486: int
    weathersit_495: int
    weathersit_513: int
    weathersit_518: int
    weathersit_571: int
    weathersit_582: int
    weathersit_623: int
    weathersit_630: int
    weathersit_640: int
    weathersit_647: int
    weathersit_659: int
    weathersit_663: int
    weathersit_672: int
    weathersit_673: int
    weathersit_674: int
    weathersit_679: int
    weathersit_684: int
    weathersit_688: int
    weathersit_692: int
    weathersit_710: int
    weathersit_731: int
    weathersit_736: int
    weathersit_773: int
    weathersit_776: int
    weathersit_789: int
    weathersit_815: int
    weathersit_817: int
    weathersit_834: int
    weathersit_836: int
    weathersit_843: int
    weathersit_844: int
    weathersit_857: int
    weathersit_859: int
    weathersit_860: int
    weathersit_870: int
    weathersit_879: int
    weathersit_893: int
    weathersit_894: int
    weathersit_899: int
    weathersit_906: int
    weathersit_907: int
    weathersit_913: int
    weathersit_919: int
    weathersit_951: int
    weathersit_965: int
    weathersit_989: int
    
    # 7 variables weathersit_XXX que faltaban en la versión 2:
    weathersit_102: int # <--- AGREGADA
    weathersit_104: int # <--- AGREGADA
    weathersit_105: int # <--- AGREGADA
    weathersit_107: int # <--- AGREGADA
    weathersit_109: int # <--- AGREGADA
    weathersit_110: int # <--- AGREGADA
    weathersit_111: int # <--- AGREGADA
    
    holiday_1: int
    workingday_1: int
    
class PredictionRequest(BaseModel):
    """Esquema para la solicitud de predicción."""
    model_name: str # Por ejemplo: ElasticNet_Casual_BikeSharing
    features: BikeSharingFeatures
