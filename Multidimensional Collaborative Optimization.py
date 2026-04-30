import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import traceback
import io
import math
import hashlib as _hl
import base64 as _b64
import plotly.graph_objects as go
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

# ===== Page Config & Constants =====
st.set_page_config(page_title="Building Retrofit Optimization Platform", page_icon="🏗️", layout="wide")

OUTPUT_EXCEL_PARETO_BASENAME = "Pareto_Solutions.xlsx"
OUTPUT_PLOT_PARETO_BASENAME = "Pareto_Front.png"
OUTPUT_EXCEL_RECOMMENDATIONS_BASENAME = "Recommended_Solutions.xlsx"
OUTPUT_PLOT_RECOMMENDATIONS_BASENAME = "Recommendation_Comparison.png"
OUTPUT_PLOT_UPGRADE_FREQ_BASENAME = "Upgrade_Frequency.png"

ELEMENT_ORDER = ['WALLT', 'WALLS', 'WINU', 'SHGC', 'SHAH', 'SHAL', 'SHAR', 'LIGHT', 'ROOFT', 'ROOFS', 'ACH']
ELEMENT_CATEGORIES = {
    'WALLT': 'Wall', 'WALLS': 'Wall', 'WINU': 'Window', 'SHGC': 'Window', 'SHAH': 'Shading', 'SHAL': 'Shading',
    'SHAR': 'Shading', 'LIGHT': 'Lighting', 'ROOFT': 'Roof', 'ROOFS': 'Roof', 'ACH': 'Ventilation'
}
BUILDING_TYPES_AVAILABLE = ["Slab", "Strip", "Y-type", "Point-type"]

GRADE_VALUES = {
    #        WALLT   WALLS  WINU  SHGC  SHAH  SHAL  SHAR  LIGHT  ROOFT  ROOFS  ACH
    'A': [0.321, 0.2, 1.65, 0.50, 0.80, 0.80, 0.80, 5,  0.31, 0.2, 0.37],
    'B': [0.301, 0.2, 3.20, 0.72, 0.60, 0.60, 0.60, 10, 0.22, 0.2, 1.10],
    'C': [0.241, 0.4, 3.70, 0.83, 0.30, 0.30, 0.30, 15, 0.15, 0.4, 1.33],
    'D': [0.163, 0.6, 4.70, 0.83, 0.12, 0.12, 0.12, 20, 0.10, 0.6, 1.80],
}
AVAILABLE_GRADES = list(GRADE_VALUES.keys())

ELEMENTS = {
    'WALLT': {'levels': [0.163, 0.241, 0.301, 0.321], 'unit': 'W/(m·K)', 'name': 'Wall Conductivity',      'icon': '🧱',    'image_prefix': 'wallt', 'higher_is_better': True},
    'WALLS': {'levels': [0.8,   0.6,   0.4,   0.2  ], 'unit': '',        'name': 'Wall Solar Absorptance', 'icon': '🧱☀️',  'image_prefix': 'walls', 'higher_is_better': False},
    'WINU':  {'levels': [4.70,  3.30,  2.40,  1.70 ], 'unit': 'W/(m²·K)','name': 'Window U-value',        'icon': '🖼️',   'image_prefix': 'winu',  'higher_is_better': False},
    'SHGC':  {'levels': [0.83,  0.72,  0.50,  0.20 ], 'unit': '',        'name': 'Window SHGC',            'icon': '🖼️☀️', 'image_prefix': 'shgc',  'higher_is_better': False},
    'SHAH':  {'levels': [0.12,  0.30,  0.60,  0.80 ], 'unit': 'm',       'name': 'Horizontal Shading',     'icon': '⛱️',   'image_prefix': 'shah',  'higher_is_better': True},
    'SHAL':  {'levels': [0.12,  0.30,  0.60,  0.80 ], 'unit': 'm',       'name': 'Left Shading',           'icon': '⛱️',   'image_prefix': 'shal',  'higher_is_better': True},
    'SHAR':  {'levels': [0.12,  0.30,  0.60,  0.80 ], 'unit': 'm',       'name': 'Right Shading',          'icon': '⛱️',   'image_prefix': 'shar',  'higher_is_better': True},
    'LIGHT': {'levels': [20,    15,    10,    5    ], 'unit': 'W/m²',    'name': 'Lighting Power Density', 'icon': '💡',   'image_prefix': 'light', 'higher_is_better': False},
    'ROOFT': {'levels': [0.10,  0.15,  0.22,  0.31 ], 'unit': 'W/(m·K)', 'name': 'Roof Conductivity',      'icon': '🏠',   'image_prefix': 'rooft', 'higher_is_better': True},
    'ROOFS': {'levels': [0.8,   0.6,   0.4,   0.2  ], 'unit': '',        'name': 'Roof Solar Absorptance', 'icon': '🏠☀️', 'image_prefix': 'roofs', 'higher_is_better': False},
    'ACH':   {'levels': [1.80,  1.33,  1.10,  0.37 ], 'unit': 'ac/h',    'name': 'Air Changes per Hour',   'icon': '💨',   'image_prefix': 'ach',   'higher_is_better': False},
}

COST_TABLE = {
    'WALLT': {0.163: 0, 0.241: 50, 0.301: 20, 0.321: 500},
    'WALLS': {0.8: 0,   0.6: 10,   0.4: 20,   0.2: 30},
    'WINU':  {4.70: 0,  3.30: 60,  2.40: 500, 1.70: 2000},
    'SHGC':  {0.83: 0,  0.72: 60,  0.50: 500, 0.20: 2000},
    'SHAH':  {0.12: 0,  0.30: 800, 0.60: 1000, 0.80: 1200},
    'SHAL':  {0.12: 0,  0.30: 300, 0.60: 500,  0.80: 1200},
    'SHAR':  {0.12: 0,  0.30: 300, 0.60: 500,  0.80: 1200},
    'LIGHT': {20: 0,    15: 100,   10: 200,    5: 400},
    'ROOFT': {0.10: 0,  0.15: 50,  0.22: 50,   0.31: 100},
    'ROOFS': {0.8: 0,   0.6: 10,   0.4: 20,    0.2: 30},
    'ACH':   {1.80: 0,  1.33: 60,  1.10: 1000, 0.37: 1500},
}

BUILDING_TYPE_MULTIPLIERS = {
    "ban": {
        'WALLT': 751.75, 'WALLS': 751.75, 'WINU': 116.85, 'SHGC': 116.85,
        'SHAH': 58, 'SHAL': 58, 'SHAR': 58, 'LIGHT': 20.0,
        'ROOFT': 206.98, 'ROOFS': 206.98, 'ACH': 296.0
    },
    "tiaoshi": {
        'WALLT': 1353.84, 'WALLS': 1353.84, 'WINU': 418.45, 'SHGC': 418.45,
        'SHAH': 209, 'SHAL': 209, 'SHAR': 209, 'LIGHT': 20.0,
        'ROOFT': 153.69, 'ROOFS': 153.69, 'ACH': 296.0
    },
    "y": {
        'WALLT': 1587.03, 'WALLS': 1587.03, 'WINU': 219.74, 'SHGC': 219.74,
        'SHAH': 150.85, 'SHAL': 150.85, 'SHAR': 150.85, 'LIGHT': 20.0,
        'ROOFT': 200.34, 'ROOFS': 200.34, 'ACH': 301.07
    },
    "dianshi": {
        'WALLT': 6262.5, 'WALLS': 6262.5, 'WINU': 1601.4, 'SHGC': 1601.4,
        'SHAH': 800, 'SHAL': 800, 'SHAR': 800, 'LIGHT': 20.0,
        'ROOFT': 767.68, 'ROOFS': 767.68, 'ACH': 296.0
    }
}

N_OBJECTIVES = 5

ELECTRICITY_PRICE = 0.8
DISCOUNT_RATE     = 0.0435
BUILDING_LIFETIME = 30

_PVA = sum(1.0 / (1.0 + DISCOUNT_RATE) ** t for t in range(1, BUILDING_LIFETIME + 1))


# 各类型楼栋总建筑面积（m²），用于 节能量(kWh/yr) = EUI差 × 面积
BUILDING_FLOOR_AREA = {
    "ban":     206.98 * 6,
    "tiaoshi": 153.69 * 6,
    "y":       200.34 * 6,
    "dianshi": 767.68 * 18,
}

# 模型输出到标准EUI(kWh/m²/yr)的换算系数
# ban/tiaoshi/y: 模型直接输出kWh/m²/yr，系数=1.0
# dianshi:       模型输出kWh/单层m²/yr，÷单层面积(767.68)得kWh/m²/yr
MODEL_EUI_FACTOR = {
    "ban":     1.0,
    "tiaoshi": 1.0,
    "y":       1.0,
    "dianshi": 1.0 / 767.68,
}



PLOT_AXIS_LABELS = {
    'Cost_per_kWh':       'Retrofit Cost per kWh Saved (¥/kWh)',
    'Average_PPD':        'Average PPD',
    'Total_Cost':         'Total Cost (¥)',
    'Energy_Saving_Rate': 'Average ESR (%)',
    'Total_ESR':          'Total ESR (kWh/m²/year)',
    'NPV':                'Net Present Value (¥)',
    'Payback_Period':     'Payback Period (years)',
}
PLOT_COLUMNS_FOR_3D = ['Cost_per_kWh', 'Average_PPD', 'Total_Cost']
IMAGE_BASE_PATH = "images"

MODEL_PATHS = {
    'ban_esr':      r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/BANSHI_lgbm_EUI_trained_model.pkl',
    'ban_ppd':      r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/BANSHI_lgbm_PPD_trained_model.pkl',
    'tiaoshi_esr':  r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/TIAOSHI_lgbm_EUI_trained_model.pkl',
    'tiaoshi_ppd':  r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/TIAOSHI_lgbm_PPD_trained_model.pkl',
    'y_esr':        r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/YSHI_lgbm_EUI_trained_model.pkl',
    'y_ppd':        r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/YSHI_lgbm_PPD_trained_model.pkl',
    'dianshi_esr':  r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/DIANSHI_lgbm_EUI_trained_model.pkl',
    'dianshi_ppd':  r'E:/BaiduSyncdisk/Shenzhen university/THESIS/moni/energyplus/AI model/DIANSHI_lgbm_PPD_trained_model.pkl',
}

# ===== Helper Functions =====
def add_log_message(message, level="info"):
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level.upper()}] {message}"
    st.session_state.log_messages.append(log_entry)
    if level in ("error", "warning"):
        print(log_entry, flush=True)

@st.cache_data
def convert_df_to_excel(df_to_convert):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_to_convert.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

@st.cache_data
def load_models_from_fixed_paths(model_paths_dict):
    models = {'ban': {}, 'tiaoshi': {}, 'y': {}, 'dianshi': {}}
    add_log_message("Loading models from fixed local paths...")
    try:
        def load_single_model(model_display_name, model_path):
            if not os.path.exists(model_path):
                add_log_message(f"'{model_display_name}' model file not found at path: {model_path}", "error")
                return None
            try:
                with open(model_path, 'rb') as f:
                    model = joblib.load(f)
                add_log_message(f"'{model_display_name}' model loaded successfully.", "info")
                return model
            except Exception as e:
                add_log_message(f"Error loading '{model_display_name}' model: {e}", "error")
                return None

        models['ban']['esr']      = load_single_model('Slab ESR',       model_paths_dict['ban_esr'])
        models['ban']['ppd']      = load_single_model('Slab PPD',        model_paths_dict['ban_ppd'])
        models['tiaoshi']['esr']  = load_single_model('Strip ESR',       model_paths_dict['tiaoshi_esr'])
        models['tiaoshi']['ppd']  = load_single_model('Strip PPD',       model_paths_dict['tiaoshi_ppd'])
        models['y']['esr']        = load_single_model('Y-type ESR',      model_paths_dict['y_esr'])
        models['y']['ppd']        = load_single_model('Y-type PPD',      model_paths_dict['y_ppd'])
        models['dianshi']['esr']  = load_single_model('Point-type ESR',  model_paths_dict['dianshi_esr'])
        models['dianshi']['ppd']  = load_single_model('Point-type PPD',  model_paths_dict['dianshi_ppd'])

        loaded_model_keys = {b: m for b, m in models.items() if m.get('esr') and m.get('ppd')}
        if not loaded_model_keys:
            add_log_message("Warning: No models were successfully loaded.", "warning")
        add_log_message("Model loading from fixed paths completed.", "success")
    except Exception as e:
        add_log_message(f"A serious error occurred while loading models: {e}", "error")
        traceback.print_exc()
        return None
    return models


def get_initial_df(building_grades_config_list):
    if not building_grades_config_list:
        add_log_message("Error: building_grades_config_list is empty.", "error")
        return pd.DataFrame(columns=ELEMENT_ORDER)
    data_for_df = []
    for i_grade, grade in enumerate(building_grades_config_list):
        if grade not in GRADE_VALUES:
            err_msg = f"Error: Grade '{grade}' (building {i_grade+1}) not defined in GRADE_VALUES."
            add_log_message(err_msg, "error")
            raise ValueError(err_msg)
        data_for_df.append(GRADE_VALUES[grade])
    return pd.DataFrame(data_for_df, columns=ELEMENT_ORDER)


def build_all_element_schemes_with_initial(initial_df, n_buildings_current):
    schemes_with_init = {}
    if initial_df.empty or n_buildings_current == 0:
        add_log_message("Warning: Initial DataFrame is empty or 0 buildings.", "warning")
        return schemes_with_init

    for elem in ELEMENT_ORDER:
        if elem not in ELEMENTS:
            add_log_message(f"Error: Element '{elem}' not in ELEMENTS.", "error")
            continue
        levels = ELEMENTS[elem]['levels']
        if not levels:
            add_log_message(f"Error: 'levels' for '{elem}' is empty.", "error")
            continue

        initial_values_elem = initial_df[elem].tolist()
        if len(initial_values_elem) != n_buildings_current:
            add_log_message(f"Warning: initial value count mismatch for '{elem}'.", "warning")
            initial_values_elem = (initial_values_elem + [levels[0]] * n_buildings_current)[:n_buildings_current]

        possible_schemes_for_elem = set()
        for target_level_idx in range(len(levels)):
            current_scheme = []
            for bldg_idx in range(n_buildings_current):
                initial_val = initial_values_elem[bldg_idx]
                try:
                    initial_idx_in_levels = levels.index(initial_val)
                except ValueError:
                    initial_idx_in_levels = min(range(len(levels)), key=lambda i: abs(levels[i] - initial_val))
                final_chosen_idx = max(initial_idx_in_levels, target_level_idx)
                current_scheme.append(levels[final_chosen_idx])
            possible_schemes_for_elem.add(tuple(current_scheme))
        schemes_with_init[elem] = sorted([list(s) for s in possible_schemes_for_elem])
    return schemes_with_init


def compute_cost(elem, new_val, bldg_idx, initial_df_local, building_type_key_local):
    if bldg_idx >= len(initial_df_local):
        add_log_message(f"compute_cost warning: bldg_idx {bldg_idx} out of range.", "warning")
        return 0

    type_multipliers = BUILDING_TYPE_MULTIPLIERS.get(building_type_key_local)
    if type_multipliers is None:
        add_log_message(f"compute_cost error: no multipliers for building type '{building_type_key_local}'.", "error")
        return float('inf')

    mult = type_multipliers.get(elem)
    if mult is None:
        add_log_message(f"compute_cost error: element '{elem}' not found in multipliers.", "error")
        return float('inf')

    init_val = initial_df_local.iloc[bldg_idx][elem]
    levels   = ELEMENTS[elem]['levels']

    def find_level_idx(val):
        for idx, lv in enumerate(levels):
            if abs(lv - val) < 1e-7:
                return idx
        nearest = min(range(len(levels)), key=lambda i: abs(levels[i] - val))
        return nearest

    init_idx   = find_level_idx(init_val)
    target_idx = find_level_idx(new_val)

    if target_idx <= init_idx:
        return 0.0

    incremental_cost = 0.0
    for step in range(init_idx + 1, target_idx + 1):
        step_val = levels[step]
        step_unit_cost = COST_TABLE[elem].get(step_val)
        if step_unit_cost is None:
            closest_key = min(COST_TABLE[elem].keys(), key=lambda k: abs(k - step_val))
            step_unit_cost = COST_TABLE[elem][closest_key]
        if elem in ('SHAH', 'SHAL', 'SHAR'):
            incremental_cost += step_unit_cost * mult * new_val
        else:
            incremental_cost += step_unit_cost * mult

    return round(incremental_cost, 2)


def calculate_metrics(features_df, models, current_building_types_list, n_buildings_current,
                      return_individual_values=False):
    results_for_df = []
    individual_esrs_list = []
    individual_ppds_list = []

    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        add_log_message("calculate_metrics: Input features_df is empty or invalid.", "warning")
        nan_list = [np.nan] * n_buildings_current
        metrics_df = pd.DataFrame({'ESR': nan_list, 'PPD': nan_list}, index=range(n_buildings_current))
        if return_individual_values:
            return metrics_df, nan_list, nan_list
        return metrics_df

    for i in range(n_buildings_current):
        esr, ppd = np.nan, np.nan
        row = features_df.iloc[i] if i < len(features_df) else None

        if row is None:
            individual_esrs_list.append(np.nan)
            individual_ppds_list.append(np.nan)
            results_for_df.append({'ESR': np.nan, 'PPD': np.nan})
            continue

        if i >= len(current_building_types_list):
            individual_esrs_list.append(np.nan)
            individual_ppds_list.append(np.nan)
            results_for_df.append({'ESR': np.nan, 'PPD': np.nan})
            continue

        btype = current_building_types_list[i]
        btype_key_map = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}
        btype_key = btype_key_map.get(btype)

        if not btype_key:
            individual_esrs_list.append(np.nan)
            individual_ppds_list.append(np.nan)
            results_for_df.append({'ESR': np.nan, 'PPD': np.nan})
            continue

        model_set = models.get(btype_key)
        if model_set is None or not model_set.get('esr') or not model_set.get('ppd'):
            individual_esrs_list.append(np.nan)
            individual_ppds_list.append(np.nan)
            results_for_df.append({'ESR': np.nan, 'PPD': np.nan})
            continue

        try:
            X   = pd.DataFrame([row[ELEMENT_ORDER].values], columns=ELEMENT_ORDER)
            esr_raw = model_set['esr'].predict(X)[0]
            ppd     = model_set['ppd'].predict(X)[0]
            # 换算为标准EUI(kWh/m²/yr)
            # ban/tiaoshi/y直接输出kWh/m²/yr；dianshi输出kWh/单层m²/yr需÷767.68
            esr = esr_raw * MODEL_EUI_FACTOR.get(btype_key, 1.0)
        except Exception as e:
            add_log_message(f"calculate_metrics PredictErr for bldg {i+1}: {e}", "error")
            esr, ppd = np.nan, np.nan

        individual_esrs_list.append(esr)
        individual_ppds_list.append(ppd)
        results_for_df.append({'ESR': esr, 'PPD': ppd})

    metrics_df = pd.DataFrame(results_for_df)
    if return_individual_values:
        return metrics_df, individual_esrs_list, individual_ppds_list
    return metrics_df


def compute_npv_and_payback(total_cost, esr_saved_per_m2, total_floor_area):
    if esr_saved_per_m2 <= 1e-9 or total_cost <= 1e-9 or total_floor_area <= 0:
        return -float('inf'), float('inf')
    annual_saving = esr_saved_per_m2 * total_floor_area * ELECTRICITY_PRICE
    if annual_saving <= 1e-9:
        return -float('inf'), float('inf')
    npv = -total_cost + annual_saving * _PVA
    pt  = total_cost / annual_saving
    return npv, pt


class CommunityRetrofitProblem(Problem):
    def __init__(self, models, initial_df, display_schemes, element_order, cd_building_indices_current,
                 current_building_types_list, n_buildings_current):
        self.models = models
        self.initial_df = initial_df
        self.display_schemes = display_schemes
        self.element_order = element_order
        self.cd_building_indices_current = cd_building_indices_current
        self.current_building_types = current_building_types_list
        self.n_buildings_current = n_buildings_current

        n_vars   = len(element_order)
        xl       = np.array([0] * n_vars)
        xu_list  = []
        for elem in element_order:
            if elem not in display_schemes or not display_schemes[elem]:
                raise ValueError(f"CommunityRetrofitProblem: element '{elem}' has no schemes.")
            xu_list.append(len(display_schemes[elem]) - 1)
        xu = np.array(xu_list)

        super().__init__(n_var=n_vars, n_obj=N_OBJECTIVES, n_constr=0, xl=xl, xu=xu, vtype=int)

        _btype_key_map = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}
        for _btype in set(current_building_types_list):
            _bkey = _btype_key_map.get(_btype)
            _ms   = self.models.get(_bkey, {}) if _bkey else {}
            if not _ms.get('esr') or not _ms.get('ppd'):
                add_log_message(f"[DIAG] Model missing for building type '{_btype}'.", "warning")
            else:
                add_log_message(f"[DIAG] Model OK for building type '{_btype}'.", "info")

        try:
            _init_metrics = calculate_metrics(
                self.initial_df.copy(), self.models,
                self.current_building_types, self.n_buildings_current
            )
            self.initial_esr_sum = _init_metrics['ESR'].sum() if not _init_metrics['ESR'].isnull().any() else None
        except Exception:
            self.initial_esr_sum = None

        try:
            _all_d_grades     = ['D'] * self.n_buildings_current
            _baseline_df      = get_initial_df(_all_d_grades)
            _baseline_metrics = calculate_metrics(
                _baseline_df, self.models,
                self.current_building_types, self.n_buildings_current
            )
            self.baseline_esr_sum = (
                _baseline_metrics['ESR'].sum()
                if not _baseline_metrics['ESR'].isnull().any() else None
            )
            self.baseline_esr_list = _baseline_metrics['ESR'].tolist()
        except Exception:
            self.baseline_esr_sum  = None
            self.baseline_esr_list = [np.nan] * self.n_buildings_current

    def _evaluate(self, x, out, *args, **kwargs):
        all_objectives = []
        if x.ndim == 1:
            x = x.reshape(1, -1)

        btype_key_map_for_cost = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}

        _diag_done       = getattr(self, '_diag_done', False)
        _cnt_decode_fail = 0
        _cnt_cd_fail     = 0
        _cnt_feat_fail   = 0
        _cnt_cost_inf    = 0
        _cnt_pred_nan    = 0
        _cnt_esr_inf     = 0
        _cnt_ok          = 0

        for i in range(x.shape[0]):
            individual_indices = x[i, :]
            params_decoded  = {}
            valid_decode    = True
            violates_cd     = False

            for elem_idx, elem in enumerate(self.element_order):
                try:
                    s_idx = int(round(individual_indices[elem_idx]))
                    if not (0 <= s_idx < len(self.display_schemes[elem])):
                        valid_decode = False
                        break
                    params_decoded[elem] = self.display_schemes[elem][s_idx]
                except Exception:
                    valid_decode = False
                    break

            if not valid_decode:
                _cnt_decode_fail += 1
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            if self.cd_building_indices_current:
                for bldg_idx_cd in self.cd_building_indices_current:
                    is_upgraded = False
                    for elem_cd in self.element_order:
                        if (bldg_idx_cd < len(params_decoded[elem_cd]) and
                                bldg_idx_cd < len(self.initial_df)):
                            if abs(params_decoded[elem_cd][bldg_idx_cd] -
                                   self.initial_df.iloc[bldg_idx_cd][elem_cd]) > 1e-9:
                                is_upgraded = True
                                break
                    if not is_upgraded:
                        violates_cd = True
                        break

            if violates_cd:
                _cnt_cd_fail += 1
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            features_list     = []
            valid_feature_gen = True
            for b_idx in range(self.n_buildings_current):
                building_features = {}
                for elem in self.element_order:
                    if elem not in params_decoded or b_idx >= len(params_decoded[elem]):
                        valid_feature_gen = False
                        break
                    building_features[elem] = params_decoded[elem][b_idx]
                if not valid_feature_gen:
                    break
                features_list.append(building_features)

            if not valid_feature_gen:
                _cnt_feat_fail += 1
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            features_df = pd.DataFrame(features_list, columns=self.element_order)

            cost_combo = 0
            for b_idx_cost in range(self.n_buildings_current):
                building_type_display = (self.current_building_types[b_idx_cost]
                                         if b_idx_cost < len(self.current_building_types) else "")
                current_type_key = btype_key_map_for_cost.get(building_type_display)
                if not current_type_key:
                    cost_combo = float('inf')
                    break
                for elem_cost in self.element_order:
                    current_val = features_df.iloc[b_idx_cost][elem_cost]
                    cost_val    = compute_cost(elem_cost, current_val, b_idx_cost,
                                               self.initial_df, current_type_key)
                    if np.isinf(cost_val):
                        cost_combo = float('inf')
                        break
                    cost_combo += cost_val
                if np.isinf(cost_combo):
                    break

            if np.isinf(cost_combo):
                _cnt_cost_inf += 1
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            metrics_df = calculate_metrics(features_df, self.models,
                                           self.current_building_types, self.n_buildings_current)

            if metrics_df['ESR'].isnull().any() or metrics_df['PPD'].isnull().any():
                _cnt_pred_nan += 1
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            retrofit_esr_sum = metrics_df['ESR'].sum()
            ppd_s            = metrics_df['PPD'].sum()

            _btype_key_map_area = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}
            # 模型直接输出EUI（kWh/m²/yr），乘总面积得整栋年能耗(kWh/yr)
            unit_floor_areas = [
                BUILDING_FLOOR_AREA.get(_btype_key_map_area.get(self.current_building_types[b], "ban"), 0)
                for b in range(self.n_buildings_current)
            ]
            total_floor_area = sum(unit_floor_areas) if sum(unit_floor_areas) > 0 else 1.0

            if self.baseline_esr_sum is None or not hasattr(self, 'baseline_esr_list'):
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            # 节能量：模型输出单位为kWh/单层m²/yr，乘单层面积得到kWh/yr
            total_energy_saved_kwh = 0.0
            valid_area_calc = True
            for b in range(self.n_buildings_current):
                b_baseline_esr = (self.baseline_esr_list[b]
                                  if b < len(self.baseline_esr_list) else np.nan)
                b_retrofit_esr = metrics_df['ESR'].iloc[b] if b < len(metrics_df) else np.nan
                if pd.isna(b_baseline_esr) or pd.isna(b_retrofit_esr):
                    valid_area_calc = False
                    break
                total_energy_saved_kwh += (b_baseline_esr - b_retrofit_esr) * unit_floor_areas[b]

            if not valid_area_calc or total_energy_saved_kwh <= 1e-6:
                cost_per_kwh = float('inf')
            else:
                cost_per_kwh = cost_combo / total_energy_saved_kwh

            total_energy_saved = self.baseline_esr_sum - retrofit_esr_sum

            core_obj = [cost_per_kwh, ppd_s, cost_combo]
            if any(pd.isna(v) or np.isinf(v) for v in core_obj):
                _cnt_esr_inf += 1
                all_objectives.append([float('inf')] * N_OBJECTIVES)
                continue

            _cnt_ok += 1

            annual_saving_eval = total_energy_saved_kwh * ELECTRICITY_PRICE
            npv, pt = compute_npv_and_payback(cost_combo, annual_saving_eval / total_floor_area, total_floor_area)

            _npv_penalty = abs(cost_combo) * 10 + 1e8
            _pt_penalty  = float(BUILDING_LIFETIME * 10)

            neg_npv = -npv if (pd.notna(npv) and not np.isinf(npv)) else _npv_penalty
            pt_safe = pt   if (pd.notna(pt)  and not np.isinf(pt))  else _pt_penalty

            current_obj = [cost_per_kwh, ppd_s, cost_combo, neg_npv, pt_safe]
            all_objectives.append(list(current_obj))

        try:
            objectives_array = np.array(all_objectives)
            if objectives_array.shape == (x.shape[0], N_OBJECTIVES):
                out["F"] = objectives_array
            else:
                out["F"] = np.full((x.shape[0], N_OBJECTIVES), float('inf'))
        except Exception as e:
            add_log_message(f"_evaluate: array conversion error: {e}", "error")
            out["F"] = np.full((x.shape[0], N_OBJECTIVES), float('inf'))

        if not _diag_done and x.shape[0] > 0:
            self._diag_done = True
            _total = x.shape[0]
            add_log_message(
                f"[DIAG _evaluate batch={_total}] "
                f"decode_fail={_cnt_decode_fail}, cd_fail={_cnt_cd_fail}, "
                f"feat_fail={_cnt_feat_fail}, cost_inf={_cnt_cost_inf}, "
                f"pred_nan={_cnt_pred_nan}, esr_inf={_cnt_esr_inf}, ok={_cnt_ok}",
                "info"
            )


ENERGY_SAVING_RATE_THRESHOLD    = 20.0
AVERAGE_PPD_THRESHOLD           = 15.0
COST_SIMILARITY_THRESHOLD_PERCENT = 3.0
ESR_SIMILARITY_THRESHOLD_ABSOLUTE = 3.0
PPD_SIMILARITY_THRESHOLD_ABSOLUTE = 3.0


def analyze_solution_focus(solution_series, initial_solution_df_for_focus, element_order_list, elements_dict):
    focus_parts = []
    categories_config = {
        "Envelope Insulation":           ['WALLT', 'WINU', 'ROOFT'],
        "Solar Radiation Control":       ['WALLS', 'SHGC', 'ROOFS', 'SHAH', 'SHAL', 'SHAR'],
        "Airtightness & Ventilation":    ['ACH'],
        "Lighting Efficiency":           ['LIGHT']
    }

    category_improvement_scores = {cat: 0.0 for cat in categories_config}
    num_buildings_in_solution = 0
    first_param_key = f'Param_{element_order_list[0]}'
    if first_param_key in solution_series and isinstance(solution_series[first_param_key], list):
        num_buildings_in_solution = len(solution_series[first_param_key])

    if num_buildings_in_solution == 0:
        return "Cannot determine number of buildings."
    if initial_solution_df_for_focus.empty or len(initial_solution_df_for_focus) != num_buildings_in_solution:
        return "Insufficient or mismatched initial data."

    total_improvement_magnitude = 0
    for b_idx in range(num_buildings_in_solution):
        for cat_name, cat_elements_list in categories_config.items():
            for elem_key in cat_elements_list:
                param_col = f'Param_{elem_key}'
                if (param_col not in solution_series or
                        not isinstance(solution_series[param_col], list) or
                        b_idx >= len(solution_series[param_col])):
                    continue
                if elem_key not in initial_solution_df_for_focus.columns:
                    continue

                initial_val = initial_solution_df_for_focus.iloc[b_idx][elem_key]
                recom_val   = solution_series[param_col][b_idx]
                if pd.isna(initial_val) or pd.isna(recom_val):
                    continue

                elem_props  = elements_dict.get(elem_key, {})
                elem_levels = elem_props.get('levels', [])
                if not elem_levels:
                    continue

                try:
                    initial_idx = elem_levels.index(initial_val)
                except ValueError:
                    initial_idx = min(range(len(elem_levels)), key=lambda i: abs(elem_levels[i] - initial_val))
                try:
                    recom_idx = elem_levels.index(recom_val)
                except ValueError:
                    recom_idx = min(range(len(elem_levels)), key=lambda i: abs(elem_levels[i] - recom_val))

                improvement = recom_idx - initial_idx
                if improvement > 0:
                    max_possible = (len(elem_levels) - 1) - initial_idx
                    if max_possible > 0:
                        normalized = improvement / max_possible
                        category_improvement_scores[cat_name] += normalized
                        total_improvement_magnitude += normalized

    if total_improvement_magnitude < 1e-3:
        return "All parameter improvements are small or no upgrades."

    sorted_categories             = sorted(category_improvement_scores.items(), key=lambda item: item[1], reverse=True)
    primary_focus_threshold       = 0.35
    secondary_focus_threshold     = 0.20

    if sorted_categories and sorted_categories[0][1] > 0:
        primary_ratio = sorted_categories[0][1] / total_improvement_magnitude
        if primary_ratio >= primary_focus_threshold:
            focus_parts.append(f"Primary focus on {sorted_categories[0][0]}")
            if len(sorted_categories) > 1 and sorted_categories[1][1] > 0:
                secondary_ratio = sorted_categories[1][1] / total_improvement_magnitude
                if secondary_ratio >= secondary_focus_threshold:
                    focus_parts.append(f"Also focuses on {sorted_categories[1][0]}")
        else:
            focus_parts.append("Comprehensive multi-faceted improvement")
            top_contributors = [cat[0] for cat in sorted_categories
                                 if total_improvement_magnitude > 0 and cat[1] / total_improvement_magnitude > 0.1]
            if top_contributors:
                focus_parts.append(f"Including: {', '.join(top_contributors[:2])}")

    return "; ".join(focus_parts) if focus_parts else "All parameter improvements are small or no upgrades."


def run_optimization_for_streamlit(population_size, n_generations, loaded_models,
                                   building_grades_list, current_building_types_list,
                                   n_buildings_current, cd_building_indices_current):
    add_log_message("Starting AGE-MOEA optimization...", "info")
    start_time = time.time()
    df_p_final_ret        = pd.DataFrame()
    fig_p_plot_ret        = None
    df_recs_excel_ret     = pd.DataFrame()
    recs_summary_dict_ret = {}
    fig_recs_comp_ret     = None
    fig_upgrade_freq_ret  = None
    st.session_state.equivalent_solutions_analysis = []

    if loaded_models is None:
        add_log_message("Error: Models not loaded.", "error")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    models = loaded_models
    try:
        initial_df_global = get_initial_df(building_grades_list)
        if initial_df_global.empty and n_buildings_current > 0:
            add_log_message("Error: Initial DataFrame is empty.", "error")
            return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False
    except ValueError as e:
        add_log_message(f"Error getting initial DataFrame: {e}", "error")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    st.session_state.initial_df_for_run = initial_df_global.copy()

    add_log_message("Calculating initial performance...", "info")
    if not initial_df_global.empty:
        _, initial_individual_esrs, initial_individual_ppds = calculate_metrics(
            initial_df_global, models, current_building_types_list,
            n_buildings_current, return_individual_values=True
        )
        initial_total_esr = sum(e for e in initial_individual_esrs if pd.notna(e))
        initial_total_ppd = sum(p for p in initial_individual_ppds if pd.notna(p))
        initial_avg_ppd   = (initial_total_ppd / n_buildings_current
                             if n_buildings_current > 0 and pd.notna(initial_total_ppd) else np.nan)
        add_log_message(
            f"Initial total ESR: {initial_total_esr:.2f}, "
            f"Initial avg PPD: {initial_avg_ppd if pd.notna(initial_avg_ppd) else 'N/A'}", "info"
        )
    else:
        initial_total_esr        = np.nan
        initial_individual_esrs  = [np.nan] * n_buildings_current

    try:
        _baseline_df_display = get_initial_df(['D'] * n_buildings_current)
        _, baseline_individual_esrs, _ = calculate_metrics(
            _baseline_df_display, models, current_building_types_list,
            n_buildings_current, return_individual_values=True
        )
        baseline_total_esr = sum(e for e in baseline_individual_esrs if pd.notna(e))
        add_log_message(f"Baseline (all-D) avg EUI: {baseline_total_esr/n_buildings_current:.2f} kWh/m²/yr", "info")
    except Exception:
        baseline_individual_esrs = [np.nan] * n_buildings_current
        baseline_total_esr       = np.nan

    constraint_ok = True
    if cd_building_indices_current and not initial_df_global.empty:
        for bldg_idx_cd in cd_building_indices_current:
            if bldg_idx_cd >= len(initial_df_global):
                constraint_ok = False
                break
            can_upgrade = False
            for elem in ELEMENT_ORDER:
                init_v  = initial_df_global.iloc[bldg_idx_cd][elem]
                lvls    = ELEMENTS[elem]['levels']
                try:
                    init_lvl_idx = lvls.index(init_v)
                except ValueError:
                    init_lvl_idx = min(range(len(lvls)), key=lambda i: abs(lvls[i] - init_v))
                if init_lvl_idx < len(lvls) - 1:
                    can_upgrade = True
                    break
            if not can_upgrade:
                constraint_ok = False
                add_log_message(
                    f"Error: C/D building {bldg_idx_cd+1} already at highest level.", "error"
                )
                break
    elif cd_building_indices_current and initial_df_global.empty:
        constraint_ok = False

    if not constraint_ok:
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    display_schemes = build_all_element_schemes_with_initial(initial_df_global, n_buildings_current)
    if not display_schemes or not all(display_schemes.get(el) for el in ELEMENT_ORDER):
        add_log_message("Error: Failed to build valid upgrade schemes.", "error")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    try:
        problem = CommunityRetrofitProblem(
            models, initial_df_global, display_schemes, ELEMENT_ORDER,
            cd_building_indices_current, current_building_types_list, n_buildings_current
        )
    except ValueError as e:
        add_log_message(f"Error initializing CommunityRetrofitProblem: {e}", "error")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    algorithm = AGEMOEA(pop_size=population_size, sampling=IntegerRandomSampling())
    add_log_message(f"Running optimization (pop: {population_size}, gen: {n_generations})...", "info")

    res = None
    try:
        res = minimize(problem, algorithm, ('n_gen', n_generations), seed=1, verbose=False)
    except Exception as e:
        add_log_message(f"Error during optimization: {e}", "error")
        traceback.print_exc()
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    if res is None or res.X is None or res.F is None or len(res.X) == 0:
        add_log_message("Error: No valid solutions found.", "error")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    add_log_message(f"Optimization completed in {time.time() - start_time:.2f}s.", "success")

    results_list = []
    _PT_PENALTY_THRESHOLD  = BUILDING_LIFETIME * 5
    _NPV_PENALTY_THRESHOLD = 1e7

    _n_total   = len(res.X)
    _n_all_inf = sum(1 for f in res.F if any(np.isinf(o) or pd.isna(o) for o in f))
    _n_core_inf = sum(1 for f in res.F if any(np.isinf(o) or pd.isna(o) for o in f[:3]))
    add_log_message(
        f"Pareto raw results: total={_n_total}, all-obj-inf={_n_all_inf}, "
        f"core-obj-inf={_n_core_inf}, potentially-valid={_n_total - _n_core_inf}", "info"
    )

    for i_res in range(len(res.X)):
        idx_res, obj_raw = res.X[i_res], res.F[i_res]
        if any(np.isinf(o) or pd.isna(o) for o in obj_raw[:3]):
            continue

        total_cost_per_kwh = obj_raw[0]
        total_ppd          = obj_raw[1]
        cost               = obj_raw[2]
        neg_npv_obj        = obj_raw[3]
        pt_obj             = obj_raw[4]

        avg_ppd = total_ppd / n_buildings_current if n_buildings_current > 0 and pd.notna(total_ppd) else np.nan

        npv_val = (-neg_npv_obj
                   if pd.notna(neg_npv_obj) and not np.isinf(neg_npv_obj) and neg_npv_obj < _NPV_PENALTY_THRESHOLD
                   else np.nan)
        pt_display = (pt_obj
                      if pd.notna(pt_obj) and not np.isinf(pt_obj) and pt_obj < _PT_PENALTY_THRESHOLD
                      else np.nan)

        current_solution_features_list = []
        valid_decode = True
        for b_idx in range(n_buildings_current):
            building_features = {}
            for el_idx, el in enumerate(ELEMENT_ORDER):
                try:
                    s_idx = int(round(idx_res[el_idx]))
                    if s_idx >= len(display_schemes[el]) or b_idx >= len(display_schemes[el][s_idx]):
                        valid_decode = False
                        break
                    building_features[el] = display_schemes[el][s_idx][b_idx]
                except Exception:
                    valid_decode = False
                    break
            if not valid_decode:
                break
            current_solution_features_list.append(building_features)

        avg_energy_saving_rate     = np.nan
        total_energy_saved_display = np.nan
        valid_sol_esrs             = []
        sol_esrs                   = [np.nan] * n_buildings_current
        if valid_decode and current_solution_features_list:
            sol_features_df = pd.DataFrame(current_solution_features_list, columns=ELEMENT_ORDER)
            _, sol_esrs, _  = calculate_metrics(
                sol_features_df, models, current_building_types_list,
                n_buildings_current, return_individual_values=True
            )
            individual_saving_rates = []
            for b_idx in range(n_buildings_current):
                base_esr = baseline_individual_esrs[b_idx] if b_idx < len(baseline_individual_esrs) else np.nan
                sol_esr  = sol_esrs[b_idx]
                if pd.notna(base_esr) and base_esr > 1e-9 and pd.notna(sol_esr):
                    individual_saving_rates.append((base_esr - sol_esr) / base_esr * 100)
                else:
                    individual_saving_rates.append(np.nan)
            valid_rates = [r for r in individual_saving_rates if pd.notna(r)]
            if valid_rates:
                avg_energy_saving_rate = sum(valid_rates) / len(valid_rates)

            valid_sol_esrs      = [e for e in sol_esrs if pd.notna(e)]
            valid_baseline_esrs = [baseline_individual_esrs[i] for i in range(n_buildings_current)
                                   if i < len(baseline_individual_esrs) and pd.notna(baseline_individual_esrs[i])]
            if valid_sol_esrs and valid_baseline_esrs and len(valid_sol_esrs) == len(valid_baseline_esrs):
                total_energy_saved_display = sum(valid_baseline_esrs) - sum(valid_sol_esrs)

        retrofit_esr_sum_display = sum(valid_sol_esrs) if valid_sol_esrs else np.nan

        _btype_key_map_disp = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}
        # 模型直接输出EUI（kWh/m²/yr），乘总面积得整栋年能耗(kWh/yr)
        unit_floor_areas_disp = [
            BUILDING_FLOOR_AREA.get(_btype_key_map_disp.get(current_building_types_list[b], "ban"), 0)
            for b in range(n_buildings_current)
        ]
        total_floor_area_disp = sum(unit_floor_areas_disp) if sum(unit_floor_areas_disp) > 0 else 1.0

        # 节能量：模型输出单位为kWh/单层m²/yr，乘单层面积得到kWh/yr
        total_energy_saved_kwh_display = np.nan
        kwh_sum   = 0.0
        valid_kwh = True
        for b in range(n_buildings_current):
            b_base = baseline_individual_esrs[b] if b < len(baseline_individual_esrs) else np.nan
            b_sol  = sol_esrs[b] if (valid_decode and b < len(sol_esrs)) else np.nan
            if pd.isna(b_base) or pd.isna(b_sol):
                valid_kwh = False
                break
            kwh_sum += (b_base - b_sol) * unit_floor_areas_disp[b]
        if valid_kwh and valid_decode:
            total_energy_saved_kwh_display = kwh_sum

        cost_per_kwh_display = (
            cost / total_energy_saved_kwh_display
            if pd.notna(cost) and pd.notna(total_energy_saved_kwh_display)
               and total_energy_saved_kwh_display > 1e-6
            else np.nan
        )

        annual_saving_display = (
            total_energy_saved_kwh_display * ELECTRICITY_PRICE
            if pd.notna(total_energy_saved_kwh_display) else np.nan
        )

        if pd.notna(annual_saving_display) and pd.notna(cost) and annual_saving_display > 1e-9 and cost > 1e-9:
            npv_val    = -cost + annual_saving_display * _PVA
            pt_display = cost / annual_saving_display
            if np.isinf(npv_val) or pd.isna(npv_val):
                npv_val = np.nan
        else:
            npv_val    = np.nan
            pt_display = np.nan

        row_res = {
            'Total_ESR':            round(retrofit_esr_sum_display, 2)      if pd.notna(retrofit_esr_sum_display)      else np.nan,
            'Total_PPD':            round(total_ppd, 2)                      if pd.notna(total_ppd)                     else np.nan,
            'Average_PPD':          round(avg_ppd, 2)                        if pd.notna(avg_ppd)                       else np.nan,
            'Total_Cost':           round(cost, 2)                           if pd.notna(cost)                          else np.nan,
            'Cost_per_kWh':         round(cost_per_kwh_display, 4)           if pd.notna(cost_per_kwh_display)          else np.nan,
            'Energy_Saving_Rate':   round(avg_energy_saving_rate, 2)         if pd.notna(avg_energy_saving_rate)        else np.nan,
            'Total_Energy_Saved':   round(total_energy_saved_kwh_display, 2) if pd.notna(total_energy_saved_kwh_display) else np.nan,
            'Annual_Saving':        round(annual_saving_display, 2)          if pd.notna(annual_saving_display)         else np.nan,
            'NPV':                  round(npv_val, 2)                        if pd.notna(npv_val)                       else np.nan,
            'Payback_Period':       round(pt_display, 2)                     if pd.notna(pt_display)                    else np.nan,
            'Scheme_Indices':       list(idx_res),
        }
        for el_idx, el in enumerate(ELEMENT_ORDER):
            try:
                s_idx = int(round(idx_res[el_idx]))
                if s_idx < len(display_schemes[el]):
                    sch_vals = display_schemes[el][s_idx]
                    row_res[f'Param_{el}'] = (sch_vals if isinstance(sch_vals, list) and len(sch_vals) == n_buildings_current
                                              else [np.nan] * n_buildings_current)
                else:
                    row_res[f'Param_{el}'] = [np.nan] * n_buildings_current
            except Exception:
                row_res[f'Param_{el}'] = [np.nan] * n_buildings_current
        results_list.append(row_res)

    if not results_list:
        add_log_message("Error: results_list is empty after processing.", "error")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, False

    df_p = pd.DataFrame(results_list)
    required_cols = ['Cost_per_kWh', 'Average_PPD', 'Total_Cost']
    df_p.dropna(subset=required_cols, inplace=True)
    df_p = df_p[np.isfinite(df_p[required_cols]).all(axis=1)]

    if df_p.empty:
        add_log_message("Warning: No valid Pareto solutions after filtering.", "warning")
        return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, True

    df_p_all = df_p.drop_duplicates(subset=required_cols).reset_index(drop=True)

    hard_constraint_mask = (
        (df_p_all['Energy_Saving_Rate'] >= ENERGY_SAVING_RATE_THRESHOLD) &
        (df_p_all['Average_PPD']        <= AVERAGE_PPD_THRESHOLD) &
        (df_p_all['NPV'] > 0)
    )
    df_p_qualified = df_p_all[hard_constraint_mask].reset_index(drop=True)

    n_total     = len(df_p_all)
    n_qualified = len(df_p_qualified)
    n_removed   = n_total - n_qualified
    add_log_message(
        f"Hard-constraint filter (ESR>={ENERGY_SAVING_RATE_THRESHOLD}%, PPD<={AVERAGE_PPD_THRESHOLD}, NPV>0): "
        f"total={n_total}, qualified={n_qualified}, removed={n_removed}", "info"
    )

    if df_p_qualified.empty:
        add_log_message("Warning: No solutions pass hard constraints. Showing full set.", "warning")
        df_p_final_ret = df_p_all
    else:
        df_p_final_ret = df_p_qualified

    add_log_message(f"Found {len(df_p_final_ret)} solutions for analysis.", "success")

    plot_df = df_p_final_ret[PLOT_COLUMNS_FOR_3D].copy()
    plot_df.dropna(inplace=True)
    plot_df = plot_df[np.isfinite(plot_df).all(axis=1)]
    if not plot_df.empty:
        try:
            fig = plt.figure(figsize=(9, 7))
            ax  = fig.add_subplot(111, projection='3d')
            sc  = ax.scatter(plot_df[PLOT_COLUMNS_FOR_3D[0]], plot_df[PLOT_COLUMNS_FOR_3D[1]],
                             plot_df[PLOT_COLUMNS_FOR_3D[2]],
                             c=plot_df[PLOT_COLUMNS_FOR_3D[2]], cmap='viridis', s=70, alpha=0.8)
            ax.set_xlabel(PLOT_AXIS_LABELS.get(PLOT_COLUMNS_FOR_3D[0]))
            ax.set_ylabel(PLOT_AXIS_LABELS.get(PLOT_COLUMNS_FOR_3D[1]))
            ax.set_zlabel(PLOT_AXIS_LABELS.get(PLOT_COLUMNS_FOR_3D[2]))
            ax.view_init(elev=20, azim=-65)
            fig.colorbar(sc, label=PLOT_AXIS_LABELS.get('Total_Cost'), ax=ax, pad=0.12)
            plt.title(
                f'Qualified Pareto Front (n={len(plot_df)})',
                fontsize=13
            )
            plt.tight_layout()
            fig_p_plot_ret = fig
        except Exception as e:
            add_log_message(f"Error generating Pareto plot: {e}", "error")

    temp_recs_plot = []
    if not df_p_final_ret.empty:
        rec_defs_primary = {
            "★ Best Economic Solution (Max NPV)": {
                "col": "NPV", "method": "idxmax",
                "desc": "★ Best Economic Solution (Max NPV)"
            },
        }
        rec_defs_secondary = {
            "Min Payback Period (Auxiliary)": {
                "col": "Payback_Period", "method": "idxmin",
                "desc": "Min Payback Period (Auxiliary)"
            },
            "Min Average PPD":            {"col": "Average_PPD",       "method": "idxmin", "desc": "Min Average PPD"},
            "Max Energy Saving Rate":     {"col": "Energy_Saving_Rate","method": "idxmax", "desc": "Max Energy Saving Rate"},
        }

        all_rec_defs = {**rec_defs_primary, **rec_defs_secondary}

        for k_rec, rdef in all_rec_defs.items():
            try:
                if rdef["col"] in df_p_final_ret.columns:
                    valid_s = df_p_final_ret[rdef["col"]][
                        np.isfinite(df_p_final_ret[rdef["col"]].fillna(float('inf')))
                        & df_p_final_ret[rdef["col"]].notna()
                    ]
                    if not valid_s.empty:
                        idx_val  = getattr(valid_s, rdef["method"])()
                        orig_idx = valid_s.index[valid_s.index == idx_val][0]
                        rec_s    = df_p_final_ret.loc[orig_idx].copy()
                        recs_summary_dict_ret[rdef["desc"]] = rec_s
                        temp_recs_plot.append(rec_s.rename(rdef["desc"]))
            except Exception as e:
                add_log_message(f"Error finding '{rdef['desc']}': {e}", "warning")

        norm_cols = PLOT_COLUMNS_FOR_3D

        def normalize_for_mcdm(source_df):
            tdf = source_df[norm_cols].copy()
            tdf.dropna(inplace=True)
            tdf = tdf[np.isfinite(tdf).all(axis=1)]
            if len(tdf) < 2:
                return None, None
            ndf = pd.DataFrame(index=tdf.index)
            for c in norm_cols:
                cmin, cmax = tdf[c].min(), tdf[c].max()
                ndf[c] = (cmax - tdf[c]) / (cmax - cmin) if abs(cmax - cmin) > 1e-9 else 1.0
            return tdf, ndf

        try:
            tdf, ndf = normalize_for_mcdm(df_p_final_ret)
            if ndf is not None:
                ideal_best  = ndf.max()
                ideal_worst = ndf.min()
                sep_best    = np.sqrt(((ndf - ideal_best)  ** 2).sum(axis=1))
                sep_worst   = np.sqrt(((ndf - ideal_worst) ** 2).sum(axis=1))
                closeness   = sep_worst / (sep_best + sep_worst + 1e-9)
                bal_s       = df_p_final_ret.loc[closeness.idxmax()].copy()
                recs_summary_dict_ret["Best Balanced Solution (TOPSIS)"] = bal_s
                temp_recs_plot.append(bal_s.rename("Best Balanced (TOPSIS)"))
        except Exception as e:
            add_log_message(f"TOPSIS error: {e}", "error")

        try:
            tdf, ndf = normalize_for_mcdm(df_p_final_ret)
            if ndf is not None:
                w     = np.array([1/3, 1/3, 1/3])
                dev   = 1.0 - ndf
                S     = (dev * w).sum(axis=1)
                R     = (dev * w).max(axis=1)
                s_min, s_max = S.min(), S.max()
                r_min, r_max = R.min(), R.max()
                v     = 0.5
                if abs(s_max - s_min) > 1e-9 and abs(r_max - r_min) > 1e-9:
                    Q = v * (S - s_min) / (s_max - s_min) + (1 - v) * (R - r_min) / (r_max - r_min)
                else:
                    Q = S
                bal_s = df_p_final_ret.loc[Q.idxmin()].copy()
                recs_summary_dict_ret["Best Balanced Solution (VIKOR)"] = bal_s
                temp_recs_plot.append(bal_s.rename("Best Balanced (VIKOR)"))
        except Exception as e:
            add_log_message(f"VIKOR error: {e}", "error")

        try:
            tdf, ndf = normalize_for_mcdm(df_p_final_ret)
            if ndf is not None:
                w     = np.array([1/3, 1/3, 1/3])
                score = (ndf * w).sum(axis=1)
                bal_s = df_p_final_ret.loc[score.idxmax()].copy()
                recs_summary_dict_ret["Best Balanced Solution (WSM)"] = bal_s
                temp_recs_plot.append(bal_s.rename("Best Balanced (WSM)"))
        except Exception as e:
            add_log_message(f"WSM error: {e}", "error")

        excel_data_list = []
        for desc, rec_s in recs_summary_dict_ret.items():
            params = {}
            for elem in ELEMENT_ORDER:
                pv = rec_s.get(f'Param_{elem}')
                params[elem] = pv if isinstance(pv, list) and len(pv) == n_buildings_current else [np.nan] * n_buildings_current

            for b_idx in range(n_buildings_current):
                btype  = current_building_types_list[b_idx] if b_idx < len(current_building_types_list) else "Unknown"
                bgrade = building_grades_list[b_idx]         if b_idx < len(building_grades_list)        else "Unknown"
                b_row  = {
                    'Recommendation_Type':  desc,
                    'Pareto_Table_Index':   rec_s.name,
                    'Building_Description': f"{btype} - Building {b_idx+1} (Initial grade: {bgrade})",
                    'Building_ID_Numeric':  b_idx + 1,
                    'Building_Type':        btype,
                    'Building_Grade_Initial': bgrade,
                    'Overall_Cost_per_kWh (¥/kWh)':            rec_s.get('Cost_per_kWh',       np.nan),
                    'Overall_Average_ESR (%)':                  rec_s.get('Energy_Saving_Rate', np.nan),
                    'Overall_Total_Energy_Saved (kWh/yr)':      rec_s.get('Total_Energy_Saved', np.nan),
                    'Overall_Annual_Saving (¥/yr)':             rec_s.get('Annual_Saving',      np.nan),
                    'Overall_NPV (¥)':                          rec_s.get('NPV',                np.nan),
                    'Overall_Payback_Period (yr)':               rec_s.get('Payback_Period',     np.nan),
                    'Overall_Average_PPD':                       rec_s.get('Average_PPD',        np.nan),
                    'Overall_Total_PPD':                         rec_s.get('Total_PPD',          np.nan),
                    'Overall_Total_Cost (¥)':                    rec_s.get('Total_Cost',         np.nan),
                    'Overall_Total_ESR (kWh/m2/yr)':             rec_s.get('Total_ESR',          np.nan),
                }
                for elem in ELEMENT_ORDER:
                    p_list = params.get(elem, [np.nan] * n_buildings_current)
                    b_row[elem] = p_list[b_idx] if b_idx < len(p_list) else np.nan
                excel_data_list.append(b_row)

        if excel_data_list:
            df_recs_excel_temp = pd.DataFrame(excel_data_list)
            cols_order = (
                ['Recommendation_Type', 'Pareto_Table_Index', 'Building_Description',
                 'Building_ID_Numeric', 'Building_Type', 'Building_Grade_Initial',
                 'Overall_Cost_per_kWh (¥/kWh)', 'Overall_Average_ESR (%)',
                 'Overall_Total_Energy_Saved (kWh/yr)', 'Overall_Annual_Saving (¥/yr)',
                 'Overall_NPV (¥)', 'Overall_Payback_Period (yr)',
                 'Overall_Average_PPD', 'Overall_Total_PPD',
                 'Overall_Total_Cost (¥)', 'Overall_Total_ESR (kWh/m2/yr)']
                + ELEMENT_ORDER
            )
            final_cols        = [c for c in cols_order if c in df_recs_excel_temp.columns]
            df_recs_excel_ret = df_recs_excel_temp[final_cols]

    if temp_recs_plot and all(isinstance(s, pd.Series) for s in temp_recs_plot):
        plot_input_df = pd.DataFrame(temp_recs_plot)
        plot_input_df.rename(columns={
            'Average_PPD':        'Average PPD',
            'Energy_Saving_Rate': 'Average ESR (%)',
            'Total_Cost':         'Total Cost (¥)',
            'NPV':                'NPV (¥)',
            'Payback_Period':     'Payback Period (yr)',
        }, inplace=True)
        try:
            fig_recs_comp_ret = plot_recommendations_comparison(plot_input_df)
        except Exception as e:
            add_log_message(f"Error generating recommendation comparison chart: {e}", "error")

    if recs_summary_dict_ret and not initial_df_global.empty:
        try:
            fig_upgrade_freq_ret = plot_upgrade_frequency(recs_summary_dict_ret, initial_df_global)
        except Exception as e:
            add_log_message(f"Error generating upgrade frequency chart: {e}", "warning")

    if not df_p_final_ret.empty:
        sorted_eq    = df_p_final_ret.copy()
        sorted_eq['original_pareto_index'] = sorted_eq.index
        sorted_eq    = sorted_eq.sort_values(by='Total_Cost').reset_index(drop=True)
        equivalent_groups = []
        processed    = set()

        for i_eq in range(len(sorted_eq)):
            cur_orig_idx = sorted_eq.loc[i_eq, 'original_pareto_index']
            if cur_orig_idx in processed:
                continue
            cur_s = sorted_eq.iloc[i_eq].copy()
            group = [cur_s]
            processed.add(cur_orig_idx)

            for j_eq in range(i_eq + 1, len(sorted_eq)):
                cand_orig_idx = sorted_eq.loc[j_eq, 'original_pareto_index']
                if cand_orig_idx in processed:
                    continue
                cand_s    = sorted_eq.iloc[j_eq].copy()
                cost_cur  = cur_s['Total_Cost']
                cost_cand = cand_s['Total_Cost']
                cost_sim  = (False if pd.isna(cost_cur) or pd.isna(cost_cand)
                             else (abs(cost_cur - cost_cand) / cost_cur * 100 <= COST_SIMILARITY_THRESHOLD_PERCENT
                                   if abs(cost_cur) >= 1e-9
                                   else abs(cost_cur - cost_cand) < 1e-9))
                cpk_sim = abs(cur_s['Cost_per_kWh'] - cand_s['Cost_per_kWh']) <= ESR_SIMILARITY_THRESHOLD_ABSOLUTE
                ppd_sim = abs(cur_s['Average_PPD']  - cand_s['Average_PPD'])  <= PPD_SIMILARITY_THRESHOLD_ABSOLUTE

                if cost_sim and cpk_sim and ppd_sim:
                    params_different = False
                    for elem in ELEMENT_ORDER:
                        lc = cur_s.get(f'Param_{elem}')
                        lx = cand_s.get(f'Param_{elem}')
                        if isinstance(lc, list) and isinstance(lx, list) and len(lc) == len(lx):
                            if lc != lx:
                                params_different = True
                                break
                        elif lc != lx:
                            params_different = True
                            break
                    if params_different:
                        group.append(cand_s)
                        processed.add(cand_orig_idx)

            if len(group) > 1:
                equivalent_groups.append(group)

        st.session_state.equivalent_solutions_analysis = equivalent_groups

    add_log_message("Optimization process completed.", "success")
    return df_p_final_ret, fig_p_plot_ret, recs_summary_dict_ret, fig_recs_comp_ret, df_recs_excel_ret, fig_upgrade_freq_ret, True


def plot_recommendations_comparison(df_recs_plot_data):
    if df_recs_plot_data.empty:
        return None

    objectives_map = {
        'Total Cost (¥)':  ['Total_Cost',         'Total Cost (¥)'],
        'Avg PPD':         ['Average_PPD',         'Average PPD'],
        'Cost/kWh (¥)':   ['Cost_per_kWh',        'Cost per kWh Saved (¥/kWh)'],
        'ESR (%)':         ['Energy_Saving_Rate',  'Average ESR (%)'],
        'NPV (¥)':         ['NPV',                 'NPV (¥)'],
        'Payback (yr)':    ['Payback_Period',       'Payback Period (yr)'],
    }

    raw_data = {}
    for display_name, potential_cols in objectives_map.items():
        for col in potential_cols:
            if col in df_recs_plot_data.columns:
                raw_data[display_name] = df_recs_plot_data[col].values
                break

    if not raw_data:
        return None

    metrics = list(raw_data.keys())
    solution_names = [str(idx) for idx in df_recs_plot_data.index]

    higher_is_better = {
        'Total Cost (¥)': False,
        'Avg PPD': False,
        'Cost/kWh (¥)': False,
        'ESR (%)': True,
        'NPV (¥)': True,
        'Payback (yr)': False,
    }

    normalized = {}
    for m in metrics:
        vals = np.array(raw_data[m], dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if abs(vmax - vmin) < 1e-9:
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - vmin) / (vmax - vmin)
        if not higher_is_better.get(m, True):
            norm = 1.0 - norm
        normalized[m] = norm

    PALETTE = [
        '#1B6CA8', '#2EAF7D', '#E8850C', '#C0392B',
        '#8E44AD', '#16A085', '#2C3E50',
    ]

    n_sol     = len(solution_names)
    n_metrics = len(metrics)
    colors    = [PALETTE[i % len(PALETTE)] for i in range(n_sol)]

    fig = plt.figure(figsize=(18, 7), facecolor='#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    ax_radar = fig.add_subplot(121, polar=True)
    ax_radar.set_facecolor('#F5F5F5')

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]

    for level in [0.25, 0.5, 0.75, 1.0]:
        circle_vals = [level] * (n_metrics + 1)
        ax_radar.plot(angles_plot, circle_vals, '-', color='#DDDDDD', linewidth=0.8, zorder=1)

    for idx, (sol_name, color) in enumerate(zip(solution_names, colors)):
        vals_norm = [normalized[m][idx] for m in metrics]
        vals_plot = vals_norm + [vals_norm[0]]
        lw  = 2.5 if idx == 0 else 1.5
        alp = 0.85 if idx == 0 else 0.65
        ax_radar.plot(angles_plot, vals_plot, 'o-', color=color,
                      linewidth=lw, markersize=5, alpha=alp, zorder=3 + idx,
                      label=sol_name)
        ax_radar.fill(angles_plot, vals_plot, color=color, alpha=0.07 if idx > 0 else 0.12)

    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(metrics, fontsize=10, fontweight='500', color='#333333')
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='#AAAAAA')
    ax_radar.set_ylim(0, 1.1)
    ax_radar.spines['polar'].set_color('#CCCCCC')
    ax_radar.grid(color='#DDDDDD', linewidth=0.5)
    ax_radar.set_title('Normalized Performance\n(higher = better for all axes)',
                        fontsize=12, fontweight='600', color='#222222', pad=18)

    ax_bar = fig.add_subplot(122)
    ax_bar.set_facecolor('#F5F5F5')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('#CCCCCC')
    ax_bar.spines['bottom'].set_color('#CCCCCC')

    key_metrics_display = ['ESR (%)', 'Avg PPD', 'Cost/kWh (¥)', 'Payback (yr)']
    key_metrics_display = [m for m in key_metrics_display if m in raw_data]

    x = np.arange(len(key_metrics_display))
    bar_width = 0.8 / max(n_sol, 1)
    offsets   = np.linspace(-(n_sol - 1) / 2, (n_sol - 1) / 2, n_sol) * bar_width

    for idx, (sol_name, color) in enumerate(zip(solution_names, colors)):
        vals = [raw_data[m][idx] for m in key_metrics_display]
        bars = ax_bar.bar(
            x + offsets[idx], vals, bar_width * 0.88,
            color=color, alpha=0.85, label=sol_name,
            edgecolor='white', linewidth=0.5, zorder=2
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                fmt = f'{v:.1f}' if abs(v) < 1000 else f'{v/1000:.1f}k'
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    fmt, ha='center', va='bottom',
                    fontsize=7.5, color='#555555', fontweight='500'
                )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(key_metrics_display, fontsize=10.5, fontweight='500', color='#333333')
    ax_bar.yaxis.set_tick_params(labelsize=8, color='#AAAAAA')
    ax_bar.set_ylabel('Value', fontsize=9, color='#666666')
    ax_bar.set_title('Key Performance Indicators\n(raw values)',
                     fontsize=12, fontweight='600', color='#222222', pad=12)
    ax_bar.grid(axis='y', color='#DDDDDD', linewidth=0.6, linestyle='--', zorder=0)
    ax_bar.set_axisbelow(True)

    handles, labels = ax_radar.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', ncol=min(n_sol, 4),
        fontsize=9, frameon=True,
        fancybox=False, edgecolor='#DDDDDD',
        facecolor='#FAFAFA',
        bbox_to_anchor=(0.5, -0.04),
        title='Solutions', title_fontsize=9,
    )

    fig.suptitle(
        f'Recommended Solutions — Performance Comparison\n'
        f'All solutions: ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%  ·  PPD ≤ {AVERAGE_PPD_THRESHOLD}  ·  NPV > 0',
        fontsize=14, fontweight='700', color='#111111', y=1.01
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def plot_upgrade_frequency(recommendations_summary, initial_df):
    if not recommendations_summary or initial_df.empty:
        return None

    upgrade_counts = {elem: 0 for elem in ELEMENT_ORDER}
    n_recs  = len(recommendations_summary)
    n_bldgs = len(initial_df)
    if n_recs == 0 or n_bldgs == 0:
        return None

    for rec_name, rec_s in recommendations_summary.items():
        for elem in ELEMENT_ORDER:
            param_list = rec_s.get(f'Param_{elem}')
            if not isinstance(param_list, list):
                continue
            for b_idx in range(n_bldgs):
                if b_idx < len(param_list) and elem in initial_df.columns:
                    init_v  = initial_df.iloc[b_idx][elem]
                    recom_v = param_list[b_idx]
                    if pd.notna(recom_v) and pd.notna(init_v):
                        try:
                            if abs(float(recom_v) - float(init_v)) > 1e-7:
                                upgrade_counts[elem] += 1
                        except (ValueError, TypeError):
                            if str(recom_v) != str(init_v):
                                upgrade_counts[elem] += 1

    total_possible = n_bldgs * n_recs
    if total_possible == 0:
        return None

    upgrade_freq = {
        ELEMENTS.get(elem, {}).get('name', elem): count / total_possible * 100
        for elem, count in upgrade_counts.items()
    }

    df_freq = pd.Series(upgrade_freq).sort_values(ascending=False)
    try:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#FAFAFA')
        ax.set_facecolor('#F5F5F5')
        n       = len(df_freq)
        cmap    = plt.cm.get_cmap('RdYlGn_r')
        colors  = [cmap(i / max(1, n - 1)) for i in range(n)]
        bars    = df_freq.plot(kind='bar', ax=ax, color=colors, width=0.72, edgecolor='white', linewidth=0.8)
        ax.set_title('Retrofit Item Implementation Frequency', fontsize=15, fontweight='700', color='#111111', pad=14)
        ax.set_ylabel('Implementation Frequency (%)', fontsize=11, color='#444444')
        ax.tick_params(axis='x', rotation=38, labelsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC', zorder=0)
        ax.set_axisbelow(True)
        for bar in bars.patches:
            y = bar.get_height()
            if pd.isna(y):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2., y + 0.8,
                    f'{y:.1f}%', ha='center', va='bottom', fontsize=8.5,
                    color='#333333', fontweight='500')
        plt.tight_layout()
    except Exception as e:
        add_log_message(f"plot_upgrade_frequency error: {e}", "error")
        return None
    return fig


def plot_parallel_coordinates_for_all_buildings(
        recommended_solution_series,
        all_pareto_solutions_df,
        initial_building_grades,
        element_order_list,
        elements_metadata):

    if all_pareto_solutions_df.empty:
        return None

    num_buildings = len(initial_building_grades)
    if num_buildings == 0:
        return None

    obj_cols = ['Cost_per_kWh', 'Energy_Saving_Rate', 'Average_PPD',
                'Total_Cost', 'NPV', 'Payback_Period']
    obj_labels = {
        'Cost_per_kWh':       'Cost/kWh\n(¥)',
        'Energy_Saving_Rate': 'ESR\n(%)',
        'Average_PPD':        'Avg.\nPPD',
        'Total_Cost':         'Total Cost\n(¥)',
        'NPV':                'NPV\n(¥)',
        'Payback_Period':     'Payback\n(yr)',
    }

    all_cols = element_order_list + obj_cols

    recommended_lines = []
    for b_idx in range(num_buildings):
        data, valid = {}, True
        for elem in element_order_list:
            col = f'Param_{elem}'
            if (col in recommended_solution_series
                    and isinstance(recommended_solution_series[col], list)
                    and len(recommended_solution_series[col]) > b_idx):
                data[elem] = recommended_solution_series[col][b_idx]
            else:
                valid = False
                break
        if valid:
            for obj in obj_cols:
                if obj in recommended_solution_series:
                    data[obj] = recommended_solution_series[obj]
            data['grade'] = initial_building_grades[b_idx]
            recommended_lines.append(data)

    if not recommended_lines:
        return None

    df_rec = pd.DataFrame(recommended_lines)

    background_lines = []
    for _, sol_row in all_pareto_solutions_df.iterrows():
        if sol_row.name == recommended_solution_series.name:
            continue
        for b_idx in range(num_buildings):
            data, valid = {}, True
            for elem in element_order_list:
                col = f'Param_{elem}'
                if (col in sol_row
                        and isinstance(sol_row[col], list)
                        and len(sol_row[col]) > b_idx):
                    data[elem] = sol_row[col][b_idx]
                else:
                    valid = False
                    break
            if valid:
                for obj in obj_cols:
                    if obj in sol_row:
                        data[obj] = sol_row[obj]
                background_lines.append(data)

    df_bg = pd.DataFrame(background_lines) if background_lines else pd.DataFrame()

    combined = (pd.concat([df_bg, df_rec], ignore_index=True)
                if not df_bg.empty else df_rec.copy())

    valid_cols = []
    for c in all_cols:
        if (c in combined.columns
                and pd.api.types.is_numeric_dtype(combined[c])
                and combined[c].notna().any()):
            valid_cols.append(c)

    if len(valid_cols) < 2:
        return None

    col_min = {}
    col_max = {}
    for c in valid_cols:
        col_min[c] = combined[c].min()
        col_max[c] = combined[c].max()
        if col_min[c] == col_max[c]:
            col_min[c] -= 1e-6
            col_max[c] += 1e-6

    def normalize_row(row):
        vals = []
        for c in valid_cols:
            v = row.get(c, np.nan)
            if pd.isna(v):
                vals.append(0.5)
            else:
                vals.append((v - col_min[c]) / (col_max[c] - col_min[c]))
        return vals

    GRADE_COLORS = {
        'A': '#2CA02C',
        'B': '#1F77B4',
        'C': '#E07B20',
        'D': '#D62728',
    }

    axis_labels = []
    for c in valid_cols:
        if c in element_order_list:
            meta = elements_metadata.get(c, {})
            name = meta.get('name', c)
            unit = meta.get('unit', '')
            lbl  = f"{name}\n({unit})" if unit else name
        else:
            lbl = obj_labels.get(c, c)
        axis_labels.append(lbl)

    n_axes = len(valid_cols)
    n_param_axes = sum(1 for c in valid_cols if c in element_order_list)

    fig_width  = max(18, n_axes * 1.3)
    fig_height = 8.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')

    x_positions = np.linspace(0, 1, n_axes)

    obj_start_idx = n_param_axes
    OBJ_HIGHLIGHT_COLOR = '#7B2FBE'
    OBJ_ALPHA           = 0.88
    OBJ_LW              = 2.8

    if not df_bg.empty:
        bg_sample = df_bg.sample(min(300, len(df_bg)), random_state=42) if len(df_bg) > 300 else df_bg
        for _, row in bg_sample.iterrows():
            y_vals = normalize_row(row)
            ax.plot(x_positions, y_vals,
                    color='#999999', alpha=0.32, linewidth=0.65,
                    solid_capstyle='round', zorder=1)

    grades_present = df_rec['grade'].dropna().unique().tolist()
    legend_handles = []

    for grade in ['D', 'C', 'B', 'A']:
        if grade not in grades_present:
            continue
        grade_color = GRADE_COLORS.get(grade, '#555555')
        df_g = df_rec[df_rec['grade'] == grade]

        for _, row in df_g.iterrows():
            y_vals = normalize_row(row)

            if n_param_axes > 0:
                x_param = x_positions[:n_param_axes]
                y_param = y_vals[:n_param_axes]
                ax.plot(x_param, y_param,
                        color=grade_color, alpha=0.92, linewidth=2.6,
                        solid_capstyle='round', zorder=3)
                ax.scatter(x_param, y_param,
                           color=grade_color, s=28, zorder=4,
                           linewidths=0.6, edgecolors='white', marker='o')

            if obj_start_idx < n_axes:
                if n_param_axes > 0 and obj_start_idx < n_axes:
                    x_bridge = [x_positions[n_param_axes - 1], x_positions[obj_start_idx]]
                    y_bridge = [y_vals[n_param_axes - 1], y_vals[obj_start_idx]]
                    ax.plot(x_bridge, y_bridge,
                            color=OBJ_HIGHLIGHT_COLOR, alpha=0.55, linewidth=1.8,
                            linestyle='--', solid_capstyle='round', zorder=3)

                x_obj = x_positions[obj_start_idx:]
                y_obj = y_vals[obj_start_idx:]
                ax.plot(x_obj, y_obj,
                        color=OBJ_HIGHLIGHT_COLOR, alpha=OBJ_ALPHA, linewidth=OBJ_LW,
                        solid_capstyle='round', zorder=3)
                ax.scatter(x_obj, y_obj,
                           color=OBJ_HIGHLIGHT_COLOR, s=38, zorder=4,
                           linewidths=0.7, edgecolors='white', marker='D')

        handle = plt.Line2D([0], [0], color=grade_color, linewidth=2.6,
                            marker='o', markersize=6,
                            markeredgecolor='white', markeredgewidth=0.6,
                            label=f'Grade {grade} — parameter axes')
        legend_handles.append(handle)

    obj_handle = plt.Line2D([0], [0], color=OBJ_HIGHLIGHT_COLOR, linewidth=OBJ_LW,
                            marker='D', markersize=6,
                            markeredgecolor='white', markeredgewidth=0.6,
                            label='Objective axes (all grades)')
    bg_handle = plt.Line2D([0], [0], color='#999999', linewidth=1.2,
                           alpha=0.7, label='All qualified Pareto solutions')
    legend_handles = [bg_handle] + legend_handles + [obj_handle]

    def fmt_val(v):
        if abs(v) >= 1e6:
            return f'{v/1e6:.1f}M'
        elif abs(v) >= 1e3:
            return f'{v/1e3:.1f}k'
        elif abs(v) < 0.01 and v != 0:
            return f'{v:.2e}'
        elif v == int(v):
            return f'{int(v)}'
        else:
            return f'{v:.2f}'

    for i, (xp, col, lbl) in enumerate(zip(x_positions, valid_cols, axis_labels)):
        is_obj_axis = (col not in element_order_list)
        axis_color  = '#4A1890' if is_obj_axis else '#333333'
        lbl_color   = '#4A1890' if is_obj_axis else '#111111'
        tick_color  = '#6A3AB0' if is_obj_axis else '#555555'

        ax.axvline(x=xp, color=axis_color,
                   linewidth=1.2 if is_obj_axis else 0.9,
                   zorder=5, alpha=0.75)

        ax.text(xp, -0.10, lbl,
                ha='center', va='top',
                fontsize=13, fontweight='700',
                color=lbl_color,
                multialignment='center',
                transform=ax.transData)

        v_min = col_min[col]
        v_max = col_max[col]

        ax.text(xp, 0.02, fmt_val(v_min),
                ha='center', va='top',
                fontsize=10, color=tick_color,
                transform=ax.transData)
        ax.text(xp, 0.98, fmt_val(v_max),
                ha='center', va='bottom',
                fontsize=10, color=tick_color,
                transform=ax.transData)

        v_mid = (v_min + v_max) / 2.0
        ax.plot([xp - 0.006, xp + 0.006], [0.5, 0.5],
                color=axis_color, linewidth=0.9, zorder=5)
        ax.text(xp + 0.014, 0.5, fmt_val(v_mid),
                ha='left', va='center',
                fontsize=9, color=tick_color,
                transform=ax.transData)

    if 0 < n_param_axes < n_axes:
        sep_x = (x_positions[n_param_axes - 1] + x_positions[n_param_axes]) / 2.0
        ax.axvline(x=sep_x, color='#AAAAAA', linewidth=1.2,
                   linestyle='--', zorder=2, alpha=0.65)
        ax.text(sep_x - 0.01, 1.11,
                '← Parameters',
                ha='right', va='bottom',
                fontsize=11, color='#888888', style='italic',
                transform=ax.transData)
        ax.text(sep_x + 0.01, 1.11,
                'Objectives →',
                ha='left', va='bottom',
                fontsize=11, color='#7B2FBE', style='italic', fontweight='600',
                transform=ax.transData)

    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.38, 1.20)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(len(legend_handles), 4),
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor='#CCCCCC',
        facecolor='white',
        handlelength=2.4,
        handletextpad=0.7,
        columnspacing=1.4,
    )
    legend.get_frame().set_linewidth(0.7)

    fig.suptitle(
        'Parallel coordinate analysis — recommended scheme vs. full qualified Pareto set',
        fontsize=14, fontweight='700', color='#111111',
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# =============================================================================
# UI LAYER
# =============================================================================

def inject_global_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    font-size: 16px !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    background: #f8f7f4 !important;
    border-right: 1px solid #e8e6df !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

.sb-section-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #999;
    margin: 1rem 0 0.45rem 0;
}

.bldg-card {
    background: #fff;
    border: 1px solid #e8e6df;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.bldg-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.bldg-label { font-size: 13px; font-weight: 500; color: #555; }

.grade-pill {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    font-size: 12px; font-weight: 600;
}
.grade-A { background:#eaf3de; color:#3b6d11; }
.grade-B { background:#e6f1fb; color:#185fa5; }
.grade-C { background:#faeeda; color:#854f0b; }
.grade-D { background:#fcebeb; color:#a32d2d; }

.cd-badge {
    font-size: 11px; padding: 2px 7px; border-radius: 99px;
    background: #fcebeb; color: #a32d2d; border: 1px solid #f7c1c1;
    margin-left: 4px;
}

.econ-table { width:100%; border-collapse:collapse; }
.econ-table td { font-size: 13px; padding: 3px 0; }
.econ-table td:last-child {
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #555;
}

.page-title {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
    line-height: 1.2;
}
.page-desc {
    font-size: 15px;
    color: #555;
    line-height: 1.7;
    max-width: 860px;
    margin-bottom: 12px;
}

.chips { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 8px; margin-bottom: 6px; }
.chip {
    font-size: 12.5px; font-weight: 500;
    padding: 4px 12px; border-radius: 99px; border: 1px solid;
    display: inline-block;
}
.chip-green { background:#eaf3de; color:#3b6d11; border-color:#c0dd97; }
.chip-blue  { background:#e6f1fb; color:#185fa5; border-color:#b5d4f4; }
.chip-amber { background:#faeeda; color:#854f0b; border-color:#fac775; }
.chip-gray  { background:#f1efe8; color:#5f5e5a; border-color:#d3d1c7; }

.sec-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #888;
    margin: 1.6rem 0 0.85rem 0;
    padding-bottom: 7px;
    border-bottom: 1px solid #eeece6;
}

.metric-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; }
.metric-card {
    flex: 1; min-width: 160px;
    background: #f8f7f4;
    border-radius: 10px;
    padding: 14px 18px;
}
.metric-card.highlight {
    background: #f0f9f4;
    border-left: 3px solid #1d9e75;
}
.m-label { font-size: 12.5px; color: #888; margin-bottom: 5px; }
.m-value {
    font-size: 26px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #111;
}
.m-unit { font-size: 13px; color: #aaa; margin-left: 4px; font-family: 'DM Sans', sans-serif; }

.styled-table-wrap {
    border: 1px solid #e8e6df;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.2rem;
}
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.styled-table th {
    background: #f8f7f4;
    font-weight: 600;
    font-size: 12px;
    color: #777;
    text-align: left;
    padding: 11px 16px;
    border-bottom: 1px solid #e8e6df;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.styled-table td {
    padding: 11px 16px;
    border-bottom: 1px solid #f0ede6;
    color: #222;
    vertical-align: middle;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: #faf9f6; }
.mono { font-family: 'DM Mono', monospace; font-size: 13px; }
.text-muted { color: #999; }

.constraint-banner {
    background: #f8f7f4;
    border: 1px solid #e8e6df;
    border-left: 3px solid #1d9e75;
    border-radius: 10px;
    padding: 13px 17px;
    font-size: 14px;
    color: #555;
    line-height: 1.65;
    margin-bottom: 1.2rem;
}
.constraint-banner strong { color: #222; font-weight: 600; }

.info-banner {
    background: #e6f1fb;
    border-left: 3px solid #378add;
    border-radius: 10px;
    padding: 13px 17px;
    font-size: 14px;
    color: #185fa5;
    line-height: 1.65;
    margin-bottom: 1.2rem;
}
.info-banner strong { font-weight: 600; }
.primary-banner {
    background: #eaf3de;
    border-left: 3px solid #1d9e75;
    border-radius: 10px;
    padding: 13px 17px;
    font-size: 14px;
    color: #3b6d11;
    line-height: 1.65;
    margin-bottom: 1.2rem;
}
.primary-banner strong { font-weight: 600; }

.constraint-row {
    display: flex; gap: 24px; flex-wrap: wrap;
    margin: 14px 0; font-size: 13.5px;
}
.c-item { display: flex; align-items: center; gap: 6px; }
.c-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.c-dot.pass { background: #1d9e75; }
.c-dot.fail { background: #e24b4a; }
.c-val { font-family: 'DM Mono', monospace; font-size: 12px; color: #555; }

.rec-section-header {
    display: flex; align-items: baseline; gap: 12px;
    margin-bottom: 18px;
}
.primary-badge {
    font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 99px;
    background: #eaf3de; color: #3b6d11;
    border: 1px solid #c0dd97;
}
.aux-badge {
    font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 99px;
    background: #e6f1fb; color: #185fa5;
    border: 1px solid #b5d4f4;
}
.ref-badge {
    font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 99px;
    background: #f1efe8; color: #5f5e5a;
    border: 1px solid #d3d1c7;
}

.upgrade-table { width:100%; border-collapse:collapse; font-size:13px; }
.upgrade-table th {
    font-weight:600; font-size:12px; color:#777; padding:9px 12px;
    text-align:left; border-bottom:1px solid #e8e6df;
    text-transform:uppercase; letter-spacing:0.04em;
}
.upgrade-table td { padding:9px 12px; border-bottom:1px solid #f0ede6; }
.upgrade-table tr:last-child td { border-bottom:none; }
.upgraded { color:#1d9e75; font-weight:600; }
.unchanged { color:#bbb; }

.empty-state {
    border: 1px dashed #ddd;
    border-radius: 10px;
    padding: 42px 20px;
    text-align: center;
    color: #bbb;
    font-size: 14px;
    margin: 8px 0;
}

.rationale-box {
    background:#faf9f6;
    border:1px solid #e8e6df;
    border-radius:10px;
    padding:14px 18px;
    font-size:13.5px;
    color:#555;
    line-height:1.7;
    margin-top:14px;
}

.log-entry {
    font-size: 12.5px;
    font-family: 'DM Mono', monospace;
    padding: 4px 10px;
    border-radius: 5px;
    margin-bottom: 4px;
    border-left: 2px solid transparent;
}
.log-info    { background:#f8f7f4; border-color:#d3d1c7; color:#555; }
.log-success { background:#eaf3de; border-color:#9fe1cb; color:#3b6d11; }
.log-warning { background:#faeeda; border-color:#fac775; color:#854f0b; }
.log-error   { background:#fcebeb; border-color:#f7c1c1; color:#a32d2d; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: #1B1B1B !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.6rem !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #000 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #e8e6df !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 10px 18px !important;
    color: #888 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 13px !important;
    font-weight: 500 !important;
}

[data-testid="stDataFrame"] {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)


def grade_pill_html(grade: str) -> str:
    cls = f"grade-{grade}" if grade in ("A", "B", "C", "D") else "grade-D"
    return f'<span class="grade-pill {cls}">{grade}</span>'


def metric_cards_html(cards: list) -> str:
    items = []
    for c in cards:
        hl = ' highlight' if c.get("highlight") else ""
        unit = f'<span class="m-unit">{c.get("unit","")}</span>' if c.get("unit") else ""
        items.append(
            f'<div class="metric-card{hl}">'
            f'  <div class="m-label">{c["label"]}</div>'
            f'  <div class="m-value">{c["value"]}{unit}</div>'
            f'</div>'
        )
    return f'<div class="metric-row">{"".join(items)}</div>'


def constraint_row_html(checks: list) -> str:
    items = []
    for c in checks:
        dot_cls = "pass" if c["pass"] else "fail"
        status  = "Pass" if c["pass"] else "Fail"
        items.append(
            f'<div class="c-item">'
            f'  <div class="c-dot {dot_cls}"></div>'
            f'  <span>{c["label"]}</span>'
            f'  <span class="c-val">{status} · {c["val"]}</span>'
            f'</div>'
        )
    return f'<div class="constraint-row">{"".join(items)}</div>'


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:4px 0 18px 0; border-bottom:1px solid #e8e6df; margin-bottom:6px;">'
            '<div style="font-size:17px;font-weight:600;letter-spacing:-0.02em;">Retrofit Platform</div>'
            '<div style="font-size:12px;color:#aaa;margin-top:3px;">AGE-MOEA · 5-objective</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-label">Buildings</div>', unsafe_allow_html=True)

        if "building_configs" not in st.session_state:
            st.session_state.building_configs = [
                {"type": BUILDING_TYPES_AVAILABLE[0], "grade": AVAILABLE_GRADES[-1], "id": 0}
            ]
            st.session_state.next_building_id = 1

        grades_now  = [c["grade"] for c in st.session_state.building_configs]
        cd_indices  = [i for i, g in enumerate(grades_now) if g in ("C", "D")]

        temp_configs   = []
        indices_remove = []

        for i, config in enumerate(st.session_state.building_configs):
            grade       = config["grade"]
            is_cd       = grade in ("C", "D")
            cd_tag      = '<span class="cd-badge">must retrofit</span>' if is_cd else ""
            pill        = grade_pill_html(grade)

            st.markdown(
                f'<div class="bldg-card">'
                f'  <div class="bldg-card-header">'
                f'    <span class="bldg-label">Building {i+1}</span>'
                f'    <div style="display:flex;align-items:center;gap:4px;">{pill}{cd_tag}</div>'
                f'  </div>',
                unsafe_allow_html=True,
            )

            col_type, col_grade, col_del = st.columns([5, 4, 1])
            type_idx  = BUILDING_TYPES_AVAILABLE.index(config["type"]) if config["type"] in BUILDING_TYPES_AVAILABLE else 0
            grade_idx = AVAILABLE_GRADES.index(config["grade"])         if config["grade"] in AVAILABLE_GRADES         else 0

            new_type  = col_type.selectbox(
                "Type", BUILDING_TYPES_AVAILABLE, index=type_idx,
                key=f"btype_{config['id']}", label_visibility="collapsed"
            )
            new_grade = col_grade.selectbox(
                "Grade", AVAILABLE_GRADES, index=grade_idx,
                key=f"bgrade_{config['id']}", label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if len(st.session_state.building_configs) > 1:
                if col_del.button("✕", key=f"del_{config['id']}", help="Remove"):
                    indices_remove.append(i)
            temp_configs.append({"type": new_type, "grade": new_grade, "id": config["id"]})

        if indices_remove:
            st.session_state.building_configs = [
                b for idx, b in enumerate(st.session_state.building_configs)
                if idx not in indices_remove
            ]
            st.rerun()
        else:
            st.session_state.building_configs = temp_configs

        if st.button("＋ Add building", key="add_bldg_btn", use_container_width=True):
            nid = st.session_state.next_building_id
            st.session_state.building_configs.append(
                {"type": BUILDING_TYPES_AVAILABLE[0], "grade": AVAILABLE_GRADES[-1], "id": nid}
            )
            st.session_state.next_building_id += 1
            st.rerun()

        st.markdown('<div class="sb-section-label">Optimization</div>', unsafe_allow_html=True)

        population_size = st.slider(
            "Population", min_value=10, max_value=500,
            value=st.session_state.get("population_size", 100), step=10,
            key="pop_size_slider"
        )
        n_generations = st.slider(
            "Generations", min_value=5, max_value=300,
            value=st.session_state.get("n_generations", 50), step=5,
            key="n_gen_slider"
        )
        st.session_state.population_size = population_size
        st.session_state.n_generations   = n_generations

        st.markdown('<div class="sb-section-label">Economics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<table class="econ-table">'
            f'<tr><td>Electricity</td><td>{ELECTRICITY_PRICE} ¥/kWh</td></tr>'
            f'<tr><td>Discount rate</td><td>{DISCOUNT_RATE*100:.2f}%</td></tr>'
            f'<tr><td>Lifetime</td><td>{BUILDING_LIFETIME} yr</td></tr>'
            f'</table>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    return population_size, n_generations


def render_page_header(n_buildings):
    st.markdown(
        f'<div class="page-title">Building Energy Retrofit Optimization</div>'
        f'<div class="page-desc">'
        f'Multi-objective optimization (AGE-MOEA) across {n_buildings} building'
        f'{"s" if n_buildings != 1 else ""}, balancing energy saving rate, thermal comfort (PPD), '
        f'retrofit cost, NPV and payback period. '
        f'Primary recommendation by max NPV; payback period as auxiliary validation.'
        f'</div>'
        f'<div class="chips">'
        f'  <span class="chip chip-green">ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%</span>'
        f'  <span class="chip chip-blue">PPD ≤ {AVERAGE_PPD_THRESHOLD}</span>'
        f'  <span class="chip chip-amber">NPV &gt; 0 Required</span>'
        f'  <span class="chip chip-gray">C/D must retrofit</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_initial_performance(loaded_models_main, current_configs):
    n_b     = len(current_configs)
    types_  = [c["type"]  for c in current_configs]
    grades_ = [c["grade"] for c in current_configs]

    st.markdown('<div class="sec-title">Initial building performance</div>', unsafe_allow_html=True)

    if loaded_models_main is None:
        st.markdown(
            '<div class="empty-state">Model files not found — check paths in MODEL_PATHS.</div>',
            unsafe_allow_html=True,
        )
        return

    type_map = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}
    for btype in set(types_):
        bkey = type_map.get(btype)
        if not bkey or not loaded_models_main.get(bkey, {}).get("esr") or not loaded_models_main.get(bkey, {}).get("ppd"):
            st.markdown(
                f'<div class="empty-state">Model for <strong>{btype}</strong> not loaded.</div>',
                unsafe_allow_html=True,
            )
            return

    try:
        init_df = get_initial_df(grades_)
        _, init_esrs, init_ppds = calculate_metrics(
            init_df, loaded_models_main, types_, n_b, return_individual_values=True
        )
        # 模型直接输出EUI（kWh/m²/yr），无需任何换算，直接显示
        valid_euis = [e for e in init_esrs if pd.notna(e)]
        valid_ppds = [p for p in init_ppds if pd.notna(p)]
        avg_eui    = sum(valid_euis) / len(valid_euis) if valid_euis else np.nan
        avg_ppd    = sum(valid_ppds) / len(valid_ppds) if valid_ppds else np.nan

        st.markdown(
            metric_cards_html([
                {"label": "Avg. EUI (kWh/m²/yr)", "value": f"{avg_eui:.2f}" if pd.notna(avg_eui) else "N/A"},
                {"label": "Avg. PPD (%)",          "value": f"{avg_ppd:.2f}" if pd.notna(avg_ppd) else "N/A"},
            ]),
            unsafe_allow_html=True,
        )

        rows_html = ""
        for i in range(n_b):
            g       = grades_[i]
            pill    = grade_pill_html(g)
            cd_tag  = '<span class="cd-badge">must retrofit</span>' if g in ("C","D") else ""
            eui_val = f"{init_esrs[i]:.2f}" if pd.notna(init_esrs[i]) else "—"
            ppd_val = f"{init_ppds[i]:.2f}" if pd.notna(init_ppds[i]) else "—"
            rows_html += (
                f"<tr>"
                f"  <td class='mono text-muted'>{'%02d' % (i+1)}</td>"
                f"  <td>{types_[i]}</td>"
                f"  <td>{pill} {cd_tag}</td>"
                f"  <td class='mono'>{eui_val}</td>"
                f"  <td class='mono'>{ppd_val}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<div class="styled-table-wrap">'
            f'<table class="styled-table">'
            f'<thead><tr>'
            f'  <th>#</th><th>Type</th><th>Grade</th>'
            f'  <th>EUI (kWh/m²/yr)</th><th>PPD (%)</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

        cd_list = [i+1 for i, g in enumerate(grades_) if g in ("C","D")]
        if cd_list:
            bldg_str = ", ".join(f"Building {x}" for x in cd_list)
            st.markdown(
                f'<div class="constraint-banner">'
                f'<strong>C/D constraint active:</strong> {bldg_str} must receive at least one '
                f'parameter upgrade. Solutions that leave these buildings unchanged are automatically discarded.'
                f'</div>',
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Error calculating initial performance: {e}")
        traceback.print_exc()


def render_recommendation_tab(key_str, rec_s, n_b_disp, types_disp, initial_df_disp):
    is_primary = key_str.startswith("★")
    is_aux     = "Auxiliary" in key_str
    is_mcdm    = any(x in key_str for x in ("TOPSIS", "VIKOR", "WSM"))

    if is_primary:
        badge = '<span class="primary-badge">Primary recommendation</span>'
    elif is_aux:
        badge = '<span class="aux-badge">Auxiliary validation</span>'
    elif is_mcdm:
        badge = '<span class="ref-badge">Multi-criteria reference</span>'
    else:
        badge = '<span class="ref-badge">Reference</span>'

    st.markdown(
        f'<div class="rec-section-header">'
        f'  <span style="font-size:16px;font-weight:600;">{key_str}</span>'
        f'  {badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if is_primary:
        st.markdown(
            '<div class="primary-banner">'
            '<strong>Max NPV selection</strong> — this solution delivers the greatest total '
            'economic return across the building lifetime, jointly accounting for upfront '
            'retrofit cost and cumulative energy-saving revenue.'
            '</div>',
            unsafe_allow_html=True,
        )

    cpk     = rec_s.get("Cost_per_kWh",      np.nan)
    avg_ppd = rec_s.get("Average_PPD",        np.nan)
    cost    = rec_s.get("Total_Cost",         np.nan)
    avg_sr  = rec_s.get("Energy_Saving_Rate", np.nan)
    saved   = rec_s.get("Total_Energy_Saved", np.nan)
    npv_v   = rec_s.get("NPV",                np.nan)
    pt_v    = rec_s.get("Payback_Period",     np.nan)
    ann_sav = rec_s.get("Annual_Saving",      np.nan)

    def _fmt(v, fmt, fallback="N/A"):
        return fmt.format(v) if pd.notna(v) and not (isinstance(v, float) and np.isinf(v)) else fallback

    st.markdown(
        metric_cards_html([
            {"label": "Cost per kWh saved",     "value": _fmt(cpk,    "{:.2f}"),      "unit": "¥/kWh"},
            {"label": "Average PPD",             "value": _fmt(avg_ppd,"{:.2f}"),      "unit": ""},
            {"label": "Total retrofit cost",     "value": _fmt(cost,   "{:,.0f}"),     "unit": "¥"},
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(
        metric_cards_html([
            {"label": "Avg ESR vs D-baseline",   "value": _fmt(avg_sr, "{:.2f}"),      "unit": "%"},
            {"label": "Energy saved (kWh/yr)",   "value": _fmt(saved,  "{:,.0f}"),     "unit": ""},
            {"label": "Annual saving",           "value": _fmt(ann_sav,"{:,.0f}"),     "unit": "¥/yr"},
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(
        metric_cards_html([
            {"label": "★ NPV (primary indicator)",
             "value": _fmt(npv_v, "{:,.0f}"), "unit": "¥", "highlight": True},
            {"label": "Payback period (auxiliary)",
             "value": _fmt(pt_v, "{:.1f}", "> lifetime"), "unit": "yr"},
        ]),
        unsafe_allow_html=True,
    )

    esr_pass = pd.notna(avg_sr)  and avg_sr  >= ENERGY_SAVING_RATE_THRESHOLD
    ppd_pass = pd.notna(avg_ppd) and avg_ppd <= AVERAGE_PPD_THRESHOLD
    pt_pass  = pd.notna(pt_v)    and not np.isinf(pt_v) and pt_v <= BUILDING_LIFETIME
    npv_pass = pd.notna(npv_v)   and not np.isinf(npv_v) and npv_v > 0

    st.markdown(
        constraint_row_html([
            {"label": f"ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%", "pass": esr_pass,
             "val": f"{avg_sr:.1f}%" if pd.notna(avg_sr) else "N/A"},
            {"label": f"PPD ≤ {AVERAGE_PPD_THRESHOLD}",         "pass": ppd_pass,
             "val": f"{avg_ppd:.2f}" if pd.notna(avg_ppd) else "N/A"},
            {"label": f"Pt ≤ {BUILDING_LIFETIME} yr",           "pass": pt_pass,
             "val": f"{pt_v:.1f} yr" if pd.notna(pt_v) and not np.isinf(pt_v) else "N/A"},
            {"label": "NPV > 0",                                 "pass": npv_pass,
             "val": f"{npv_v:,.0f} ¥" if pd.notna(npv_v) and not np.isinf(npv_v) else "N/A"},
        ]),
        unsafe_allow_html=True,
    )

    _rationale_map = {
        "★ Best Economic Solution (Max NPV)":
            "Selected by maximizing NPV from the pool of qualified solutions "
            f"(ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%, PPD ≤ {AVERAGE_PPD_THRESHOLD}, and NPV > 0). "
            f"NPV integrates upfront cost and {BUILDING_LIFETIME}-year discounted savings "
            f"(rate {DISCOUNT_RATE*100:.2f}%), making it the most comprehensive single economic indicator. "
            "Cross-check with the payback period tab for capital recovery speed.",
        "Min Payback Period (Auxiliary)":
            "Shortest static payback period among qualified candidates. "
            "Compare with the Max NPV solution: if both point to the same solution, "
            "confidence is high. If they differ, weigh project-specific cash-flow requirements.",
        "Min Average PPD":
            "Lowest average predicted percentage dissatisfied (PPD) — best thermal comfort. "
            "Useful when occupant well-being is the overriding goal.",
        "Max Energy Saving Rate":
            "Highest energy saving rate vs the all-D baseline. "
            "Relevant when meeting regulatory ESR targets is the primary driver.",
        "Best Balanced Solution (TOPSIS)":
            "TOPSIS multi-criteria selection balancing Cost/kWh, Average PPD, and Total Cost "
            "with equal weights. All input solutions satisfy the hard constraints.",
        "Best Balanced Solution (VIKOR)":
            "VIKOR method balancing group utility and individual regret. "
            "Robust when stakeholder preferences are heterogeneous.",
        "Best Balanced Solution (WSM)":
            "Weighted Sum Method with equal weights — simple, transparent, "
            "easy to justify in reports.",
    }
    rationale = _rationale_map.get(
        key_str,
        f"Selected from the qualified solution pool "
        f"(ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%, PPD ≤ {AVERAGE_PPD_THRESHOLD}, and NPV > 0)."
    )
    st.markdown(
        f'<div class="rationale-box">{rationale}</div>',
        unsafe_allow_html=True,
    )

    if not initial_df_disp.empty and len(initial_df_disp) == n_b_disp:
        st.markdown(
            '<div style="font-size:12px;font-weight:600;color:#777;'
            'letter-spacing:0.05em;text-transform:uppercase;margin:18px 0 10px 0;">'
            'Parameter changes by building</div>',
            unsafe_allow_html=True,
        )

        header_cells = "<th>Element</th>"
        for b in range(n_b_disp):
            header_cells += f"<th>{types_disp[b]}&nbsp;{b+1}</th>"

        rows_html = ""
        for elem in ELEMENT_ORDER:
            props    = ELEMENTS.get(elem, {})
            elem_lbl = f"{props.get('name', elem)}"
            unit_lbl = f"<br><span style='font-size:11px;color:#bbb;'>{props.get('unit','')}</span>" if props.get('unit') else ""
            row      = f"<td>{elem_lbl}{unit_lbl}</td>"
            for b in range(n_b_disp):
                init_v  = initial_df_disp.iloc[b].get(elem, np.nan)
                p_list  = rec_s.get(f"Param_{elem}")
                if isinstance(p_list, list) and b < len(p_list):
                    rec_v = p_list[b]
                    if pd.notna(rec_v) and pd.notna(init_v) and abs(float(rec_v) - float(init_v)) > 1e-7:
                        row += f"<td class='upgraded mono'>{rec_v:.3f} ↑</td>"
                    else:
                        row += "<td class='unchanged'>—</td>"
                else:
                    row += "<td class='text-muted'>N/A</td>"
            rows_html += f"<tr>{row}</tr>"

        st.markdown(
            f'<div class="styled-table-wrap">'
            f'<table class="upgrade-table">'
            f'<thead><tr>{header_cells}</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    # ── Parallel coordinates section ────────────────────────────────────────
    st.markdown(
        '<div style="font-size:12px;font-weight:600;color:#777;'
        'letter-spacing:0.05em;text-transform:uppercase;margin:18px 0 10px 0;">'
        'Parameter space — parallel coordinates</div>',
        unsafe_allow_html=True,
    )
    try:
        fig_pcp = plot_parallel_coordinates_for_all_buildings(
            rec_s,
            st.session_state.get("df_pareto", pd.DataFrame()),
            st.session_state.run_specific_config.get("grades", []),
            ELEMENT_ORDER,
            ELEMENTS,
        )
        if fig_pcp:
            # ── Step 1: Save bytes FIRST, then close the figure ──────────────
            _pcp_buf = io.BytesIO()
            fig_pcp.savefig(
                _pcp_buf, format="png", dpi=200,
                bbox_inches="tight", facecolor="white"
            )
            _pcp_bytes = _pcp_buf.getvalue()
            plt.close(fig_pcp)  # close after saving, before rendering

            # ── Step 2: Display using st.image (bytes, not figure object) ────
            st.image(_pcp_bytes, use_container_width=True)

            # ── Step 3: Unique key via md5 hash (avoids truncation collisions) ─
            _safe_key = _hl.md5(key_str.encode("utf-8")).hexdigest()[:10]

            _col_dl, _col_cp, _col_sp = st.columns([1, 1, 4])

            with _col_dl:
                st.download_button(
                    label="⬇ Download PNG",
                    data=_pcp_bytes,
                    file_name="parallel_coordinates.png",
                    mime="image/png",
                    key=f"dl_pcp_{_safe_key}",
                    use_container_width=True,
                )

            with _col_cp:
                # ── Step 4: Robust JS copy — atob decode avoids quote issues ─
                _b64_str = _b64.b64encode(_pcp_bytes).decode("utf-8")
                # Split the b64 string into JS concat chunks to avoid any
                # single-quote / backslash issues inside the onclick attribute.
                # We store it as a JS variable in a <script> tag keyed by safe_key.
                _copy_btn_id = f"cp_btn_{_safe_key}"
                _script_id   = f"cp_script_{_safe_key}"
                st.markdown(
                    f"""
<button id="{_copy_btn_id}"
    style="width:100%;padding:0.38rem 0.6rem;font-size:13px;
           font-weight:500;border-radius:8px;cursor:pointer;
           background:transparent;border:1px solid #d0cec8;
           color:#444;line-height:1.4;"
    onclick="copyPcpImage_{_safe_key}()">⎘ Copy image</button>

<script id="{_script_id}">
function copyPcpImage_{_safe_key}() {{
    var b64 = "{_b64_str}";
    var byteChars = atob(b64);
    var byteNums  = new Array(byteChars.length);
    for (var i = 0; i < byteChars.length; i++) {{
        byteNums[i] = byteChars.charCodeAt(i);
    }}
    var byteArr = new Uint8Array(byteNums);
    var blob    = new Blob([byteArr], {{type: "image/png"}});
    navigator.clipboard.write([new ClipboardItem({{"image/png": blob}})])
        .then(function() {{
            var btn = document.getElementById("{_copy_btn_id}");
            if (btn) {{ btn.innerText = "✓ Copied!"; }}
            setTimeout(function() {{
                var btn2 = document.getElementById("{_copy_btn_id}");
                if (btn2) {{ btn2.innerText = "⎘ Copy image"; }}
            }}, 1800);
        }})
        .catch(function(err) {{
            alert("Copy failed: " + err);
        }});
}}
</script>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="empty-state">Not enough Pareto solutions to render parallel coordinates.</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        add_log_message(f"Parallel coords error: {e}", "warning")


def render_logs():
    msgs = st.session_state.get("log_messages", [])
    has_error = any("[ERROR]" in m for m in msgs)
    with st.expander("Run logs", expanded=has_error):
        if not msgs:
            st.markdown('<div class="empty-state">No logs yet.</div>', unsafe_allow_html=True)
            return
        parts = []
        for msg in reversed(msgs):
            if   "[ERROR]"   in msg: cls = "log-error"
            elif "[WARNING]" in msg: cls = "log-warning"
            elif "[SUCCESS]" in msg: cls = "log-success"
            else:                    cls = "log-info"
            parts.append(f'<div class="log-entry {cls}">{msg}</div>')
        st.markdown(
            f'<div style="max-height:380px;overflow-y:auto;">'
            f'{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# MAIN ENTRY
# =============================================================================

def setup_matplotlib_chinese_font():
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"]       = 12
        plt.rcParams["axes.titlesize"]  = 14
        plt.rcParams["axes.labelsize"]  = 12
        plt.rcParams["xtick.labelsize"] = 11
        plt.rcParams["ytick.labelsize"] = 11
        plt.rcParams["legend.fontsize"] = 11
    except Exception as e:
        add_log_message(f"Warning setting Matplotlib fonts: {e}", "warning")


setup_matplotlib_chinese_font()
inject_global_css()

# ── Session defaults ──────────────────────────────────────────────────────────
default_session_state_keys = {
    "optimization_run_key":          0,
    "df_pareto":                     pd.DataFrame(),
    "fig_pareto":                    None,
    "recommendations_summary":       {},
    "fig_recs_comparison":           None,
    "df_recommendations_excel":      pd.DataFrame(),
    "log_messages":                  [],
    "loaded_models_cache":           None,
    "cd_violation_logged_count":     0,
    "fig_upgrade_freq":              None,
    "run_specific_config":           None,
    "initial_df_for_run":            pd.DataFrame(),
    "equivalent_solutions_analysis": [],
}
for _k, _v in default_session_state_keys.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

population_size, n_generations = render_sidebar()

num_buildings_intro = len(st.session_state.get("building_configs", []))
render_page_header(num_buildings_intro)

st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
loaded_models_main = load_models_from_fixed_paths(MODEL_PATHS)

if not st.session_state.building_configs:
    st.markdown(
        '<div class="empty-state">Add at least one building in the sidebar to begin.</div>',
        unsafe_allow_html=True,
    )
else:
    render_initial_performance(loaded_models_main, st.session_state.building_configs)

st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
run_clicked = st.button(
    "Run optimization",
    type="primary",
    use_container_width=True,
    key="start_optimization_button",
)

if run_clicked:
    st.session_state.optimization_run_key += 1
    st.session_state.log_messages = []
    for _k in ["df_pareto", "fig_pareto", "recommendations_summary",
               "fig_recs_comparison", "df_recommendations_excel",
               "fig_upgrade_freq", "run_specific_config",
               "initial_df_for_run", "equivalent_solutions_analysis"]:
        st.session_state[_k] = default_session_state_keys[_k]
    st.session_state.cd_violation_logged_count = 0

    if not st.session_state.building_configs:
        st.error("Please configure at least one building.")
        st.stop()

    current_configs_run = st.session_state.building_configs
    n_b_run    = len(current_configs_run)
    types_run  = [c["type"]  for c in current_configs_run]
    grades_run = [c["grade"] for c in current_configs_run]
    cd_indices = [i for i, g in enumerate(grades_run) if g in ("C", "D")]

    st.session_state.run_specific_config = {
        "types": list(types_run), "grades": list(grades_run), "count": n_b_run
    }

    st.session_state.loaded_models_cache = load_models_from_fixed_paths(MODEL_PATHS)
    if st.session_state.loaded_models_cache is None:
        st.error("Model loading failed.")
        st.stop()

    type_map_run  = {"Slab": "ban", "Strip": "tiaoshi", "Y-type": "y", "Point-type": "dianshi"}
    missing_types = [
        t for t in set(types_run)
        if not type_map_run.get(t)
        or not st.session_state.loaded_models_cache.get(type_map_run[t], {}).get("esr")
        or not st.session_state.loaded_models_cache.get(type_map_run[t], {}).get("ppd")
    ]
    if missing_types:
        st.error(f"Models for {missing_types} failed to load.")
        st.stop()

    with st.spinner(f"Running optimization — {n_b_run} buildings, pop {population_size}, gen {n_generations} …"):
        df_p_res, fig_p_res, rec_sum, fig_comp, df_xl, fig_freq, success = run_optimization_for_streamlit(
            population_size, n_generations,
            st.session_state.loaded_models_cache,
            grades_run, types_run, n_b_run, cd_indices,
        )

    if success:
        st.session_state.df_pareto               = df_p_res
        st.session_state.fig_pareto              = fig_p_res
        st.session_state.recommendations_summary = rec_sum
        st.session_state.fig_recs_comparison     = fig_comp
        st.session_state.df_recommendations_excel = df_xl
        st.session_state.fig_upgrade_freq        = fig_freq

        if df_p_res is not None and not df_p_res.empty:
            st.balloons()
            st.success("Optimization complete — results below.")
        elif df_p_res is not None and df_p_res.empty:
            st.warning("Optimization finished but no qualified Pareto solutions found.")
        else:
            st.error("Optimization returned no data.")
    else:
        st.error("Optimization failed — see logs for details.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.optimization_run_key > 0 and st.session_state.run_specific_config:
    run_cfg         = st.session_state.run_specific_config
    n_b_disp        = run_cfg["count"]
    types_disp      = run_cfg["types"]
    grades_disp     = run_cfg["grades"]
    initial_df_disp = st.session_state.get("initial_df_for_run", pd.DataFrame())

    if st.session_state.recommendations_summary:
        st.markdown('<div class="sec-title">Recommended retrofit solutions</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-banner">'
            f'All solutions shown have passed hard constraints: '
            f'<strong>ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%</strong>, '
            f'<strong>PPD ≤ {AVERAGE_PPD_THRESHOLD}</strong>, and '
            f'<strong>NPV > 0</strong>. '
            f'The <strong>★ Best Economic Solution</strong> tab is the primary recommendation '
            f'(max NPV). All other tabs are reference alternatives.'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.fig_recs_comparison:
            try:
                st.pyplot(st.session_state.fig_recs_comparison)
                buf = io.BytesIO()
                st.session_state.fig_recs_comparison.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                st.download_button(
                    "Download comparison chart",
                    buf.getvalue(), OUTPUT_PLOT_RECOMMENDATIONS_BASENAME, "image/png",
                    key="dl_recs_comp_plot",
                )
                plt.close(st.session_state.fig_recs_comparison)
            except Exception as e:
                st.error(f"Chart error: {e}")

        rec_keys = [str(k) for k in st.session_state.recommendations_summary.keys()]
        if rec_keys:
            tabs = st.tabs(rec_keys)
            for i_tab, key_str in enumerate(rec_keys):
                rec_s = None
                for k, v in st.session_state.recommendations_summary.items():
                    if str(k) == key_str:
                        rec_s = v
                        break
                if rec_s is None:
                    continue
                with tabs[i_tab]:
                    render_recommendation_tab(key_str, rec_s, n_b_disp, types_disp, initial_df_disp)

        if st.session_state.df_recommendations_excel is not None and \
                not st.session_state.df_recommendations_excel.empty:
            st.download_button(
                "Download full recommendations (Excel)",
                convert_df_to_excel(st.session_state.df_recommendations_excel),
                OUTPUT_EXCEL_RECOMMENDATIONS_BASENAME,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_recs_excel_button",
            )

        if st.session_state.fig_upgrade_freq:
            st.markdown('<div class="sec-title">Retrofit item frequency</div>', unsafe_allow_html=True)
            try:
                st.pyplot(st.session_state.fig_upgrade_freq)
                buf = io.BytesIO()
                st.session_state.fig_upgrade_freq.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                st.download_button(
                    "Download frequency chart",
                    buf.getvalue(), OUTPUT_PLOT_UPGRADE_FREQ_BASENAME, "image/png",
                    key="dl_freq_plot",
                )
                plt.close(st.session_state.fig_upgrade_freq)
            except Exception as e:
                st.error(f"Chart error: {e}")

    elif st.session_state.optimization_run_key > 0:
        st.markdown(
            '<div class="empty-state">No qualified solutions found. Try larger population/generation settings.</div>',
            unsafe_allow_html=True,
        )

    eq_groups = st.session_state.get("equivalent_solutions_analysis", [])
    if eq_groups is not None:
        st.markdown('<div class="sec-title">Equivalent solution analysis</div>', unsafe_allow_html=True)
        if not eq_groups:
            st.markdown(
                f'<div class="empty-state">'
                f'No equivalent groups found (ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%, '
                f'PPD ≤ {AVERAGE_PPD_THRESHOLD}, NPV > 0).'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"Groups with similar objectives but different retrofit measures — "
                f"all already satisfy ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%, PPD ≤ {AVERAGE_PPD_THRESHOLD}, and NPV > 0."
            )
            for i_grp, group in enumerate(eq_groups):
                with st.expander(f"Equivalent group {i_grp+1}  ({len(group)} solutions)"):
                    summary_rows = []
                    for sol_s in group:
                        focus = (
                            "N/A" if initial_df_disp.empty
                            else analyze_solution_focus(sol_s, initial_df_disp, ELEMENT_ORDER, ELEMENTS)
                        )
                        summary_rows.append({
                            "Index":            sol_s.get("original_pareto_index", sol_s.name),
                            "ESR (%)":          sol_s.get("Energy_Saving_Rate", np.nan),
                            "PPD":              sol_s.get("Average_PPD",        np.nan),
                            "Total Cost (¥)":   sol_s.get("Total_Cost",         np.nan),
                            "NPV (¥)":          sol_s.get("NPV",                np.nan),
                            "Pt (yr)":          sol_s.get("Payback_Period",     np.nan),
                            "Design focus":     focus,
                        })
                    st.dataframe(pd.DataFrame(summary_rows).set_index("Index"), use_container_width=True)

    if st.session_state.df_pareto is not None and not st.session_state.df_pareto.empty:
        st.markdown('<div class="sec-title">Qualified Pareto front</div>', unsafe_allow_html=True)
        st.caption(
            f"{len(st.session_state.df_pareto)} solutions — "
            f"ESR ≥ {ENERGY_SAVING_RATE_THRESHOLD}%, PPD ≤ {AVERAGE_PPD_THRESHOLD}, and NPV > 0."
        )

        if st.session_state.fig_pareto:
            try:
                st.pyplot(st.session_state.fig_pareto)
                buf = io.BytesIO()
                st.session_state.fig_pareto.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                st.download_button(
                    "Download Pareto chart",
                    buf.getvalue(), OUTPUT_PLOT_PARETO_BASENAME, "image/png",
                    key="dl_pareto_plot",
                )
                plt.close(st.session_state.fig_pareto)
            except Exception as e:
                st.error(f"Chart error: {e}")

        disp_cols = [
            c for c in [
                "Energy_Saving_Rate", "Average_PPD", "Total_Cost",
                "Total_ESR", "Annual_Saving", "NPV", "Payback_Period",
            ]
            if c in st.session_state.df_pareto.columns
        ]
        if disp_cols:
            st.dataframe(st.session_state.df_pareto[disp_cols].head(10), use_container_width=True)
            if len(st.session_state.df_pareto) > 10:
                with st.expander(f"View all {len(st.session_state.df_pareto)} solutions"):
                    st.dataframe(st.session_state.df_pareto[disp_cols], use_container_width=True)

        st.download_button(
            "Download Pareto set (Excel)",
            convert_df_to_excel(st.session_state.df_pareto),
            OUTPUT_EXCEL_PARETO_BASENAME,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_pareto_excel_button",
        )

if st.session_state.optimization_run_key > 0:
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    render_logs()