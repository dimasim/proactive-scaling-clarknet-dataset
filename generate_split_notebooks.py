import nbformat as nbf

# ==========================================
# Notebook 1: Data Processing & EDA
# ==========================================
nb1 = nbf.v4.new_notebook()
cells1 = []
cells1.append(nbf.v4.new_markdown_cell("""# 🚀 Prediksi Workload: Tahap 1 - Preprocessing & Feature Engineering
Notebook ini mendemonstrasikan tahap awal data science pipeline: pembersihan data, *resampling*, normalisasi, dan pembuatan tensor 3D untuk model sequence.
Data akan disimpan pada akhirnya agar dapat dimuat oleh model-model secara independen."""))

cells1.append(nbf.v4.new_markdown_cell("""## Latar Belakang & Evaluasi Dataset
Data mentah jarang siap pakai. Tahap ini meliputi penanganan data hilang, duplikasi, outlier, dan standardisasi.
Membandingkan deret waktu mentah vs agregasi 2-detik untuk mengurangi fluktuasi.

### Tabel 2: Worldcup98 dataset (Referensi Imdoukh et al.)
| Ref. | Dataset | Size | Usage |
|---|---|---|---|
| S1 | Workload per second | 5,244,199 | Evaluation |
| S2 | Workload per second | 2,255,274 | Evaluation |
| M1 | Workload per minute | 87,710 | Training |
| M2 | Workload per minute | 37,590 | Evaluation |"""))

cells1.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Menangani data hilang, menghapus duplikasi
df_raw = pd.read_csv('all_merged_metrics.csv')
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='s')
df_raw.set_index('timestamp', inplace=True)
df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
df_raw.interpolate(method='time', inplace=True)
df_raw.bfill(inplace=True)

# 2. Resampling 1 Menit (mengambil nilai maksimum) sesuai Imdoukh
df_resampled = df_raw.resample('1min').max().dropna()

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)

split_raw = int(len(df_raw) * 0.70)
split_res = int(len(df_resampled) * 0.70)

# Plot 1: Raw RPS Media
axes[0].plot(df_raw.index[:split_raw], df_raw['rps_media'].iloc[:split_raw], label='S1', color='tab:blue')
axes[0].plot(df_raw.index[split_raw:], df_raw['rps_media'].iloc[split_raw:], label='S2', color='tab:orange')
axes[0].set_title('Fig. 8.1 gasss-predict dataset represented as workload per second (Media Service)', fontsize=12)
axes[0].set_ylabel('Workload (HTTP Requests)')
axes[0].legend()

# Plot 2: Resampled RPS Media
axes[1].plot(df_resampled.index[:split_res], df_resampled['rps_media'].iloc[:split_res], label='M1', color='tab:blue')
axes[1].plot(df_resampled.index[split_res:], df_resampled['rps_media'].iloc[split_res:], label='M2', color='tab:orange')
axes[1].set_title('Fig. 9.1 Simplified dataset represented as max workload per minute (Media Service)', fontsize=12)
axes[1].set_ylabel('Workload (HTTP Requests)')
axes[1].legend()

# Plot 3: Raw RPS Content
axes[2].plot(df_raw.index[:split_raw], df_raw['rps_content'].iloc[:split_raw], label='S1', color='tab:blue')
axes[2].plot(df_raw.index[split_raw:], df_raw['rps_content'].iloc[split_raw:], label='S2', color='tab:orange')
axes[2].set_title('Fig. 8.2 gasss-predict dataset represented as workload per second (Content Service)', fontsize=12)
axes[2].set_ylabel('Workload (HTTP Requests)')
axes[2].legend()

# Plot 4: Resampled RPS Content
axes[3].plot(df_resampled.index[:split_res], df_resampled['rps_content'].iloc[:split_res], label='M1', color='tab:blue')
axes[3].plot(df_resampled.index[split_res:], df_resampled['rps_content'].iloc[split_res:], label='M2', color='tab:orange')
axes[3].set_title('Fig. 9.2 Simplified dataset represented as max workload per minute (Content Service)', fontsize=12)
axes[3].set_ylabel('Workload (HTTP Requests)')
axes[3].legend()

plt.tight_layout()
plt.show()

# 3. Menangani Outlier & Normalisasi
Q1 = df_resampled['cpu_media'].quantile(0.25)
Q3 = df_resampled['cpu_media'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df_resampled['cpu_media'] = np.where(df_resampled['cpu_media'] > upper_bound, upper_bound, df_resampled['cpu_media'])

cpu_scaler = MinMaxScaler(feature_range=(-1, 1))
df_resampled['cpu_media_scaled'] = cpu_scaler.fit_transform(df_resampled[['cpu_media']])

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df_resampled[['cpu_media', 'rps_media', 'ram_media']])
df_scaled = pd.DataFrame(scaled_data, columns=['cpu_media', 'rps_media', 'ram_media'], index=df_resampled.index)
"""))

cells1.append(nbf.v4.new_markdown_cell("""## Feature Engineering
Mengekstrak fitur *Time Steps*, menggunakan 120 time steps ke belakang sebagai memory memori penuh."""))

cells1.append(nbf.v4.new_code_cell("""features = ['cpu_media', 'rps_media', 'ram_media']
df_features = df_scaled[features]

def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), :])
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)

TIME_STEPS = 10
X, y = create_dataset(df_features.values, TIME_STEPS)
print(f"Total Sampel Data 3D: {X.shape[0]} | Shape X Tensor: {X.shape} | Shape Y Tensor: {y.shape}")

# Split Dataset (70% Train, 30% Test, lalu 10% dari Train untuk Validation)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, shuffle=False)

# Simpan data ke disk agar model bisa mengaksesnya
np.savez_compressed('processed_data.npz', 
                    X_train=X_train, y_train=y_train, 
                    X_val=X_val, y_val=y_val, 
                    X_test=X_test, y_test=y_test)
with open('cpu_scaler.pkl', 'wb') as f:
    pickle.dump(cpu_scaler, f)
print("Berhasil menyimpan data latih dan scaler.")
"""))

nb1['cells'] = cells1
with open('01_Data_Processing_and_EDA.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb1, f)


# ==========================================
# Helper to generate Model Notebooks
# ==========================================
def create_model_notebook(model_name, model_def_code):
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell(f"""# Pelatihan Model & Evaluasi: {model_name}
Di sini kita memuat data preprocessed dan melatih model {model_name} raksasa."""))
    
    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Bidirectional, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load preprocessed data
data = np.load('processed_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
TIME_STEPS = X_train.shape[1]
input_shape = (TIME_STEPS, 3)

with open('cpu_scaler.pkl', 'rb') as f:
    cpu_scaler = pickle.load(f)
"""))
    
    cells.append(nbf.v4.new_markdown_cell(f"""## Arsitektur {model_name}"""))
    
    cells.append(nbf.v4.new_code_cell(model_def_code))
    
    cells.append(nbf.v4.new_code_cell(f"""early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
csv_logger = CSVLogger('training_progress_{model_name}.csv', append=True)
EPOCHS = 50
BATCH_SIZE = 64

print(f"\\n--- Training: {model_name} ---")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop, csv_logger], verbose=1)
"""))
    
    cells.append(nbf.v4.new_markdown_cell(f"""## Evaluasi Kecepatan & Akurasi"""))
    
    cells.append(nbf.v4.new_code_cell(f"""single_sample = X_test[0:1]

# Pemanasan prediksi
_ = model.predict(single_sample, verbose=0)

# Precise inference benchmark
t_arr = []
for _ in range(30):
    t_s = time.time()
    model.predict(single_sample, verbose=0)
    t_arr.append(time.time() - t_s)
avg_inference_ms = np.mean(t_arr) * 1000

# Prediksi seluruh data test
pred = model.predict(X_test, verbose=0)
np.save('pred_{model_name}.npy', pred.flatten())

results = {{
    '{model_name}': {{
        'MSE': mean_squared_error(y_test, pred.flatten()),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred.flatten())),
        'MAE': mean_absolute_error(y_test, pred.flatten()),
        'R_Squared (R²)': r2_score(y_test, pred.flatten()),
        'Speed (ms)': avg_inference_ms
    }}
}}
display(pd.DataFrame(results).T)
"""))
    nb['cells'] = cells
    with open(f'0{list(models.keys()).index(model_name) + 2}_Model_{model_name}.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


models = {
    "LSTM": """model = Sequential([
    LSTM(30, activation='tanh', input_shape=input_shape),
    Dense(1)
], name="LSTM_Imdoukh")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()""",
    "GRU": """model = Sequential([
    GRU(30, activation='tanh', input_shape=input_shape),
    Dense(1)
], name="GRU_Imdoukh")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()""",
    "BiLSTM": """model = Sequential([
    Bidirectional(LSTM(30, activation='tanh'), input_shape=input_shape),
    Dense(1)
], name="BiLSTM_Imdoukh")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()""",
    "DUCFF": """inputs = Input(shape=input_shape)
x_tcn = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='causal', activation='relu')(inputs)
x_tcn = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='causal', activation='relu')(x_tcn)
x_tcn_flat = tf.keras.layers.Flatten()(x_tcn)
attention_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
x_transformer_flat = tf.keras.layers.Flatten()(attention_output)
fusion = tf.keras.layers.Concatenate()([x_tcn_flat, x_transformer_flat])
x = Dense(256, activation='relu')(fusion)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)
model = Model(inputs=inputs, outputs=outputs, name="DUCFF_Monster")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()"""
}

for m, code in models.items():
    create_model_notebook(m, code)


# ==========================================
# Notebook 6: Autoscaler Simulation
# ==========================================
nb6 = nbf.v4.new_notebook()
cells6 = []
cells6.append(nbf.v4.new_markdown_cell("""# 🤖 Evaluasi Kinerja Auto-Scaler
Meniru evaluasi elastisitas Auto-Scaler. Kami memuat hasil prediksi yang dihasilkan masing-masing model untuk disimulasikan."""))

cells6.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Load Real Test Data & Scaler
data = np.load('processed_data.npz')
y_test = data['y_test']

with open('cpu_scaler.pkl', 'rb') as f:
    cpu_scaler = pickle.load(f)

# Load Predictions if available
model_names = ['LSTM', 'GRU', 'BiLSTM', 'DUCFF']
predictions = {}
for name in model_names:
    if os.path.exists(f'pred_{name}.npy'):
        predictions[name] = np.load(f'pred_{name}.npy')
        print(f"Loaded predictions for {name}")
    else:
        print(f"File pred_{name}.npy not found. Skip.")
"""))

cells6.append(nbf.v4.new_code_cell("""y_test_real = cpu_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
CPU_CAPACITY_PER_CONTAINER = 2.0
R_min = 5
CDT_MAX = 10  
SDR = 0.40  

def simulate_autoscaler(predicted_cpu, actual_cpu):
    R_current = R_min; CDT = 0; supply_history = []; demand_history = []
    for t in range(len(predicted_cpu)):
        R_demand = max(int(np.ceil(actual_cpu[t] / CPU_CAPACITY_PER_CONTAINER)), R_min)
        demand_history.append(R_demand)
        R_estimated = max(int(np.ceil(predicted_cpu[t] / CPU_CAPACITY_PER_CONTAINER)), R_min)
        
        if R_estimated > R_current:
            R_current = R_estimated
            CDT = CDT_MAX
        elif R_estimated < R_current:
            if CDT <= 0:
                CDT = CDT_MAX
                R_current = max(R_current - int(np.floor((R_current - R_estimated) * (1 - SDR))), R_estimated, R_min)
            else:
                CDT -= 1
        else:
            if CDT > 0: CDT -= 1
        supply_history.append(R_current)
    return np.array(supply_history), np.array(demand_history)

total_T = len(y_test_real)
demand_real = np.array([max(int(np.ceil(d / CPU_CAPACITY_PER_CONTAINER)), R_min) for d in y_test_real])
reactive_supply = np.zeros(total_T)
reactive_supply[0] = R_min
for t in range(1, total_T):
    reactive_supply[t] = max(int(np.ceil(y_test_real[t-1] / CPU_CAPACITY_PER_CONTAINER)), R_min)

def calc_metrics(supply, demand):
    under = np.maximum(demand - supply, 0)
    over = np.maximum(supply - demand, 0)
    safe_demand = np.where(demand == 0, 1, demand)
    return ((100.0/total_T)*np.sum(under/safe_demand), (100.0/total_T)*np.sum(over/safe_demand), (np.sum(under>0)/total_T)*100.0, (np.sum(over>0)/total_T)*100.0)

theta_U_n, theta_O_n, T_U_n, T_O_n = calc_metrics(reactive_supply, demand_real)
autoscaler_metrics = {'Reactive': {'θ_U': theta_U_n, 'θ_O': theta_O_n, 'T_U': T_U_n, 'T_O': T_O_n, 'η (Speedup)': 1.00}}

all_supplies = {}
if len(predictions) > 0:
    for name, pred in predictions.items():
        pred_real = cpu_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        supply, demand = simulate_autoscaler(pred_real, y_test_real)
        all_supplies[name] = supply
        t_U_a, t_O_a, TU_a, TO_a = calc_metrics(supply, demand)
        eta = ((theta_U_n/t_U_a if t_U_a>0 else 1)*(theta_O_n/t_O_a if t_O_a>0 else 1)*(T_U_n/TU_a if TU_a>0 else 1)*(T_O_n/TO_a if TO_a>0 else 1))**0.25
        autoscaler_metrics[name] = {'θ_U': t_U_a, 'θ_O': t_O_a, 'T_U': TU_a, 'T_O': TO_a, 'η (Speedup)': eta}

    display(pd.DataFrame(autoscaler_metrics).T)
else:
    print("Prediksi belum ada, silahkan jalankan notebook model terlebih dahulu.")
"""))

cells6.append(nbf.v4.new_markdown_cell("""## Total Started Containers (Overhead / API Oscillations)"""))

cells6.append(nbf.v4.new_code_cell("""if len(all_supplies) > 0:
    started_containers = {}
    for name, supply in all_supplies.items():
        started_containers[name] = sum((supply[i] - supply[i-1]) for i in range(1, len(supply)) if supply[i] > supply[i-1])

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(started_containers.keys()), y=list(started_containers.values()), palette='coolwarm')
    plt.title('Total Containers Started (Overhead)', fontsize=14)
    plt.show()
"""))

cells6.append(nbf.v4.new_markdown_cell("""## Qualitative Analysis of Demand vs Supply (GDS Algorithm)"""))

cells6.append(nbf.v4.new_code_cell("""if len(all_supplies) > 0:
    fig, axes = plt.subplots(len(all_supplies), 1, figsize=(14, 5 * len(all_supplies)), sharex=True)
    plot_len = 400
    
    # Kalo cuma ada 1 model, axes mungkin bukan list, jadi kita akali:
    if len(all_supplies) == 1:
        axes = [axes]

    for i, (m_name, supply) in enumerate(all_supplies.items()):
        axes[i].plot(demand_real[:plot_len], label='Demand (Red)', color='red', linestyle='--')
        axes[i].plot(supply[:plot_len], label=f'Supply {m_name} (Blue)', color='blue', drawstyle='steps-post', alpha=0.7)
        axes[i].set_title(f'({m_name}): Auto-Scaler Supply vs Demand', fontsize=12)
        axes[i].legend()

    plt.tight_layout()
    plt.show()
"""))

nb6['cells'] = cells6
with open('06_Auto_Scaler_Simulation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb6, f)

print("Berhasil membuat file generate_split_notebooks.py! Sekarang run file ini untuk menghasilkan notebook-notebook baru.")
