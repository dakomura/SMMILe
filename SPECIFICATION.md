# SMMILe Fork: Trident出力対応 ユーザーガイド

## 概要

tridentで抽出したCONCHv1.5特徴量（h5形式）をSMMILeで使用できるようにするためのガイドです。

### このフォークでできること

- tridentが出力するh5ファイルをSMMILe形式に変換（symlinkベース）
- スキャン倍率（40x/20x）の自動検出と座標変換
- 混在スキャン倍率のWSIでも統一された処理が可能
- 独自データセットでの学習（コード変更不要）

### 追加・修正されたファイル

| ファイル | 説明 |
|---------|------|
| `create_smmile_structure.py` | symlink作成スクリプト（自動推定機能付き） |
| `superpixel_generation.py` | h5対応・座標変換追加 |
| `single/datasets/dataset_nic.py` | h5対応・座標変換追加 |
| `multi/datasets/dataset_nic.py` | h5対応・座標変換追加 |
| `single/main.py` | 独自データセット対応（csv_path, label_dict） |
| `single/configs_custom/config_custom_template.yaml` | 独自データセット用設定テンプレート |

---

## クイックスタート

trident出力からSMMILeで学習を開始するまでの基本的な流れです。

### Step 1: ディレクトリ構造の変換

```bash
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15
```

### Step 2: スーパーピクセルの生成

```bash
python superpixel_generation.py \
    --fea_dir smmile_data/features_conch_v15 \
    --sp_dir smmile_data/superpixels_n16_c50_512 \
    --file_suffix '*.h5' \
    --n_segments_persp 16 \
    --compactness 50
```

### Step 3: CSVファイルの準備

```csv
case_id,slide_id,label
patient_001,slide_001,tumor
patient_002,slide_002,normal
```

### Step 4: 設定ファイルの作成

```bash
cd single
cp configs_custom/config_custom_template.yaml configs_custom/config_my_dataset.yaml
# config_my_dataset.yaml を編集
```

### Step 5: 学習の実行

```bash
python main.py --config configs_custom/config_my_dataset.yaml
```

---

## 1. Symlinkスクリプトの使い方

`create_smmile_structure.py` はtridentの出力ディレクトリからSMMILeが期待するディレクトリ構造をsymlinkで作成します。

### 1.1 基本的な使用方法

```bash
# 推奨：自動推定を使用
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15
```

### 1.2 ドライラン（確認用）

実際にsymlinkを作成せず、処理内容のみ確認：

```bash
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15 \
    --dry_run --verbose
```

### 1.3 固定値を使用する場合

自動推定を無効化して固定値を指定：

```bash
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15 \
    --no_auto_estimate \
    --patch_level 0 \
    --patch_size 512
```

### 1.4 スライドリストで絞り込み

特定のスライドのみ処理：

```bash
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15 \
    --slide_list slide_list.csv
```

### 1.5 コマンドラインオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--trident_features_dir` | (必須) | tridentの特徴量ディレクトリパス |
| `--output_dir` | (必須) | SMMILe用の出力ディレクトリパス |
| `--patch_level` | None | パッチレベル（自動推定時は不要） |
| `--patch_size` | None | パッチサイズ（自動推定時は不要） |
| `--no_auto_estimate` | False | 自動推定を無効化 |
| `--slide_list` | None | 処理対象スライドのリストファイル |
| `--file_pattern` | `*.h5` | 処理するファイルのパターン |
| `--dry_run` | False | 実際にsymlinkを作成しない |
| `--skip_validation` | False | h5ファイルの検証をスキップ |
| `--verbose` | False | 詳細な出力を表示 |

### 1.6 出力されるファイル名

- 入力: `TCGA-XXX.h5`
- 出力: `TCGA-XXX_{patch_level}_{patch_size}.h5`

例：
- 40xスキャン → `TCGA-XXX_1_512.h5`
- 20xスキャン → `TCGA-YYY_0_512.h5`

---

## 2. Superpixel生成

SMMILeのInstance Samplingに必要なスーパーピクセルを生成します。

### 2.1 基本的な使用方法

```bash
python superpixel_generation.py \
    --fea_dir smmile_data/features_conch_v15 \
    --sp_dir smmile_data/superpixels_n16_c50_512 \
    --file_suffix '*.h5' \
    --n_segments_persp 16 \
    --compactness 50
```

### 2.2 40x/20x混在データの処理

スキャン倍率が混在していても自動で座標変換されます：

```bash
python superpixel_generation.py \
    --fea_dir smmile_data/features_conch_v15 \
    --file_suffix '*.h5' \
    --n_segments_persp 16 \
    --compactness 50
```

### 2.3 npyファイルの処理（従来形式）

```bash
python superpixel_generation.py \
    --fea_dir path/to/npy/features \
    --file_suffix '*_1_512.npy' \
    --keyword_feature feature \
    --n_segments_persp 16 \
    --compactness 50
```

---

## 3. 独自データセットでの学習

コードを変更することなく、独自のデータセットで学習できます。

### 3.1 CSVファイルの形式

以下の3カラムが必須です：

```csv
case_id,slide_id,label
patient_001,slide_001_a,tumor
patient_001,slide_001_b,tumor
patient_002,slide_002,normal
patient_003,slide_003,tumor
```

| カラム | 説明 |
|-------|------|
| `case_id` | 患者ID（同一患者の複数スライドをグループ化） |
| `slide_id` | スライドID（特徴量ファイル名と一致させる） |
| `label` | ラベル（文字列、label_dictで数値に変換） |

### 3.2 設定ファイルの作成

テンプレートをコピーして編集：

```bash
cd single
cp configs_custom/config_custom_template.yaml configs_custom/config_my_dataset.yaml
```

必須の設定項目：

```yaml
# クラス数
n_classes: 2

# タスク名（任意の名前）
task: 'my_custom_task'

# CSVファイルのパス
csv_path: 'dataset_csv/my_dataset.csv'

# ラベル定義（文字列 → 数値のマッピング）
label_dict:
  normal: 0
  tumor: 1

# パス設定
data_root_dir: '/path/to/features/'
data_sp_dir: '/path/to/superpixels/'
data_mag: '0_512'
patch_size: 512
fea_dim: 768  # CONCHv1.5の場合
```

### 3.3 学習の実行

```bash
python main.py --config configs_custom/config_my_dataset.yaml
```

### 3.4 ファイル命名規則

| ファイル種別 | 命名規則 |
|------------|---------|
| 特徴量 | `{slide_id}_{data_mag}.h5` または `.npy` |
| スーパーピクセル | `{slide_id}_{data_mag}.npy` |

### 3.5 後方互換性

既存のタスク名（`camelyon`, `lung_subtype`など）を使用する場合は従来通りの動作をします。

---

## 4. Configファイル設定リファレンス

### 4.1 タスク設定

| パラメータ | 型 | 説明 | 例 |
|-----------|-----|------|-----|
| `n_classes` | int | 分類クラス数 | `2` |
| `task` | str | タスク名（識別子） | `'tumor_detection'` |
| `multi_label` | bool | マルチラベル分類 | `False` |

### 4.2 パス設定

| パラメータ | 型 | 説明 | 例 |
|-----------|-----|------|-----|
| `data_root_dir` | str | 特徴量ディレクトリ | `'/path/to/features/'` |
| `data_sp_dir` | str | スーパーピクセルディレクトリ | `'/path/to/superpixels/'` |
| `models_dir` | str/None | 事前学習済みモデルのパス | `None` |
| `data_mag` | str | ファイル名識別子（level_patchsize） | `'0_512'` |
| `patch_size` | int | パッチサイズ（ピクセル） | `512` |
| `fea_dim` | int | 特徴量次元数（CONCHv1.5=768） | `768` |
| `results_dir` | str | 結果保存先 | `'./results/'` |
| `exp_code` | str | 実験識別コード | `'exp_v1'` |
| `csv_path` | str | 独自データセット用CSV | `'dataset_csv/my_data.csv'` |
| `label_dict` | dict | ラベルマッピング | `{normal: 0, tumor: 1}` |

### 4.3 学習設定

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `seed` | int | `1` | 乱数シード |
| `k` | int | `5` | 交差検証のfold数 |
| `k_start` | int | `0` | 開始fold |
| `k_end` | int | `5` | 終了fold |
| `max_epochs` | int | `50` | 最大エポック数 |
| `lr` | float | `0.00002` | 学習率 |
| `reg` | float | `0.00001` | L2正則化 |
| `opt` | str | `'adam'` | 最適化アルゴリズム |
| `bag_loss` | str | `'bce'` | 損失関数（`'bce'`/`'ce'`/`'bibce'`） |
| `early_stopping` | bool | `True` | 早期終了 |
| `drop_out` | bool | `True` | ドロップアウト |
| `drop_rate` | float | `0.25` | ドロップアウト率 |
| `weighted_sample` | bool | `True` | 重み付きサンプリング |

### 4.4 モデル設定

| パラメータ | 型 | 説明 | 例 |
|-----------|-----|------|-----|
| `model_type` | str | モデルタイプ | `'smmile'` |
| `model_size` | str | `'small'`=256次元、`'big'`=512次元 | `'small'` |

### 4.5 SMMILe独自モジュール

#### Consistency Constraint（論文Fig. 3a）

Negative bagに対してattention分布の一貫性を強制。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `consistency` | bool | `False` | 有効化 |

#### Instance Dropout（論文Fig. 3c）

高スコアのパッチをランダムにドロップ。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `drop_with_score` | bool | `False` | 有効化 |
| `D` | int | `10` | ドロップアウト回数 |

#### Delocalized Instance Sampling（論文Fig. 3d）

スーパーピクセルベースの空間的に分散したサンプリング。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `superpixel` | bool | `False` | 有効化 |
| `G` | int | `10` | サンプリング回数 |
| `sp_smooth` | bool | `True` | 境界の平滑化 |

#### MRF-based Instance Refinement（論文Fig. 3e）

パッチレベル予測の洗練化。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `inst_refinement` | bool | `False` | 有効化 |
| `inst_rate` | float | `0.1` | インスタンス選択率 |
| `n_refs` | int | `3` | リファインメント層数 |
| `ref_start_epoch` | int | `0` | 開始エポック |
| `mrf` | bool | `False` | MRF制約の有効化 |
| `tau` | float | `0.1` | MRF制約の重み |

### 4.6 推奨設定パターン

#### Stage 1: ベースライン

```yaml
consistency: False
drop_with_score: False
superpixel: False
inst_refinement: False
mrf: False
```

#### Stage 2: 空間認識強化（推奨）

```yaml
consistency: True
drop_with_score: True
D: 10
superpixel: True
G: 10
sp_smooth: True
inst_refinement: False
mrf: False
```

#### Stage 3: Instance Refinement（2段階学習）

```yaml
models_dir: '/path/to/stage2/model/'
consistency: True
drop_with_score: True
D: 10
superpixel: True
G: 10
sp_smooth: True
inst_refinement: True
inst_rate: 0.1
n_refs: 3
ref_start_epoch: 0
mrf: True
tau: 0.1
```

### 4.7 完全な設定例

```yaml
# ============================================
# タスク設定
# ============================================
n_classes: 2
task: 'tumor_detection'

# ============================================
# パス設定
# ============================================
data_root_dir: '/data/features/conch_v15/'
data_sp_dir: '/data/superpixels/n16_c50_512/'
data_mag: '0_512'
patch_size: 512
fea_dim: 768
results_dir: './results/'
exp_code: 'tumor_detection_v1'
models_dir: null
label_frac: 0.8

# 独自データセット用
csv_path: 'dataset_csv/my_dataset.csv'
label_dict:
  normal: 0
  tumor: 1

# ============================================
# 学習設定
# ============================================
seed: 1
k: 5
k_start: 0
k_end: 5
max_epochs: 50
lr: 0.00002
reg: 0.00001
opt: 'adam'
bag_loss: 'bce'
early_stopping: True
drop_out: True
drop_rate: 0.25
weighted_sample: True
log_data: True
testing: False
reverse_train_val: False

# ============================================
# モデル設定
# ============================================
model_type: 'smmile'
model_size: 'small'

# ============================================
# SMMILe独自モジュール
# ============================================
consistency: True
drop_with_score: True
D: 10
superpixel: True
G: 10
sp_smooth: True
inst_refinement: False
inst_rate: 0.1
n_refs: 3
ref_start_epoch: 0
mrf: False
tau: 0.1
```

---

## 5. トラブルシューティング

### 5.1 よくあるエラー

#### `KeyError: 'features'` または `KeyError: 'coords'`

**原因**: h5ファイルの構造が期待と異なる

**確認方法**:
```python
import h5py
with h5py.File('file.h5', 'r') as f:
    print(list(f.keys()))
```

#### 座標間隔がpatch_sizeと一致しない

**原因**: 座標変換が正しく行われていない

**確認方法**:
```python
import h5py
with h5py.File('file.h5', 'r') as f:
    print(dict(f['coords'].attrs))
```

#### symlink作成失敗

**原因**: ファイルシステムがsymlinkをサポートしていない

**対処**: 絶対パスでsymlinkを作成するか、ファイルをコピー

### 5.2 デバッグモード

```bash
# superpixel_generation.pyのデバッグ出力
python superpixel_generation.py \
    --fea_dir <DIR> \
    --debug

# テストでの詳細確認
python tests/test_trident_integration.py --h5_path <FILE>
```

---

## 6. Heatmap生成（Instance Prediction可視化）

学習済みモデルを使用して、パッチレベルの予測確率をWSI上にオーバーレイしたheatmapを生成できます。

### 6.1 概要

1. `eval.py`でインスタンスレベルの予測結果（`*_inst.csv`）を生成
2. `generate_heatmap.py`でWSI上にheatmapをオーバーレイ

### 6.2 Step 1: Instance Predictionの実行

```bash
cd single
python eval.py \
  --data_root_dir /path/to/features/ \
  --data_sp_dir /path/to/superpixels/ \
  --results_dir /path/to/training/results/ \
  --models_exp_code <experiment_code>_s1 \
  --save_exp_code _heatmap \
  --split test \
  --k 5 \
  --k_start 0 \
  --k_end 5
```

#### 主要パラメータ

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `--data_root_dir` | 特徴量ディレクトリ | `/path/to/features/` |
| `--data_sp_dir` | スーパーピクセルディレクトリ | `/path/to/superpixels/` |
| `--results_dir` | 学習結果のディレクトリ | `/path/to/results/` |
| `--models_exp_code` | 実験コード（`_s1`付き） | `esca_subtyping_smmile_1512_5fold_s1` |
| `--save_exp_code` | 保存用サフィックス | `_heatmap` |
| `--split` | 対象データ | `train`, `val`, `test`, `all` |

#### 出力ファイル

`eval_results/EVAL_{models_exp_code}{save_exp_code}/`に以下が生成されます：
- `fold_X.csv` - スライドレベルの予測結果
- `smmile_inst_fold_X_inst.csv` - インスタンスレベルの予測結果

`*_inst.csv`の形式：
```csv
filename,label,prob,pred,prob_0,prob_1
TCGA-XXX/0_19456_2048.png,-1,0.0001,0,0.0001,0.00001
```

| カラム | 説明 |
|-------|------|
| `filename` | パッチの位置（`{slide_id}/{x}_{y}_{patch_size}.png`） |
| `label` | インスタンスラベル（なければ-1） |
| `prob` | 予測クラスの確率 |
| `pred` | 予測ラベル |
| `prob_X` | 各クラスの確率 |

### 6.3 Step 2: Heatmap生成

```bash
cd /path/to/SMMILe
python generate_heatmap.py \
  --model_name smmile \
  --wsi_dir "/path/to/wsi/*.svs" \
  --results_dir ./single/eval_results/EVAL_{models_exp_code}{save_exp_code} \
  --class_labels "0:class0_name,1:class1_name" \
  --coord_scale 2.0 \
  --num_workers 8
```

#### 主要パラメータ

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `--model_name` | モデル名（ファイル名に使用） | `smmile` |
| `--wsi_dir` | WSIファイルのパス（glob形式） | `"/data/wsi/*.svs"` |
| `--results_dir` | `*_inst.csv`があるディレクトリ | `./eval_results/EVAL_xxx` |
| `--class_labels` | クラスラベル | `"0:adenocarcinoma,1:squamous"` |
| `--coord_scale` | 座標スケーリング係数 | `2.0` |
| `--num_workers` | 並列処理ワーカー数 | `8` |

#### coord_scaleについて

CSVの座標がWSI Level 0座標と異なる場合にスケーリングが必要です。

| データソース | 座標スケール | coord_scale |
|-------------|-------------|-------------|
| Level 0座標 | 1:1 | `1.0` |
| Level 1相当 | 1:2 | `2.0` |
| Level 2相当 | 1:4 | `4.0` |

座標の確認方法：
```python
import pandas as pd
import openslide

# CSVの座標最大値
df = pd.read_csv('*_inst.csv')
print("Max X:", df['filename'].str.extract(r'/(\d+)_')[0].astype(int).max())
print("Max Y:", df['filename'].str.extract(r'_(\d+)_\d+\.png')[0].astype(int).max())

# WSIのサイズ
slide = openslide.OpenSlide('slide.svs')
print("WSI dimensions:", slide.level_dimensions[0])

# ratio = WSI_size / CSV_max → coord_scale
```

### 6.4 出力ファイル

`{results_dir}/visual/`に以下が生成されます：
- `{slide_id}_{model_name}_{class_name}_heatmap.png`

各クラスごとに別々のheatmapファイルが生成されます。

### 6.5 実行例（ESCA）

```bash
# Step 1: テストデータのInstance Prediction
cd /home/is1kd/work/SMMILe/single
python eval.py \
  --data_root_dir /home/is1kd/work/TITAN/TCGA/SMMILe/features/ \
  --data_sp_dir /home/is1kd/work/TITAN/TCGA/SMMILe/superpixels_n16_c50_512 \
  --results_dir /home/is1kd/work/TITAN/TCGA/SMMILe/ESCA/results/ \
  --models_exp_code esca_subtyping_smmile_1512_5fold_s1 \
  --save_exp_code _esca_test \
  --split test \
  --k 5 --k_start 0 --k_end 5

# Step 2: Heatmap生成
cd /home/is1kd/work/SMMILe
python generate_heatmap.py \
  --model_name smmile \
  --wsi_dir "/home/is1kd/work/apply_paget/TCGA/ESCA/*.svs" \
  --results_dir ./single/eval_results/EVAL_esca_subtyping_smmile_1512_5fold_s1_esca_test \
  --class_labels "0:adenocarcinoma,1:squamous_cell_carcinoma" \
  --coord_scale 2.0 \
  --num_workers 8
```

### 6.6 splitオプションの使い分け

| split | 説明 | 用途 |
|-------|------|------|
| `train` | 訓練データのみ | 学習データの確認 |
| `val` | 検証データのみ | ハイパーパラメータ調整 |
| `test` | テストデータのみ | 最終評価・可視化 |
| `all` | 全データ | 全スライドの可視化 |

---

## 7. 技術仕様

### 7.1 座標変換

tridentの座標はWSI level 0座標で保存されています。スキャン倍率によって座標間隔が異なるため、読み込み時に変換が必要です。

| スキャン倍率 | ターゲット倍率 | patch_size | 座標間隔（level 0） |
|------------|--------------|------------|-------------------|
| 40x | 20x | 512 | 1024 |
| 20x | 20x | 512 | 512 |

**変換処理**:

```python
def load_features_and_coords_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        coords_level0 = f['coords'][:]
        
        # 属性から倍率情報を取得
        attrs = dict(f['coords'].attrs)
        patch_size = attrs.get('patch_size', 512)
        level0_mag = attrs.get('level0_magnification', 40)
        target_mag = attrs.get('target_magnification', 20)
        
        # 座標をターゲット倍率座標に変換
        scale_factor = target_mag / level0_mag
        coords_nd = (coords_level0 * scale_factor).astype(np.int64)
    
    return features, coords_nd, patch_size
```

**変換後の結果**:

```
40x scan: [4096, 19456] → [2048, 9728] (step: 1024 → 512)
20x scan: [2048, 9728]  → [2048, 9728] (step: 512 → 512)
```

### 7.2 自動推定機能

h5ファイルの属性から`patch_level`と`patch_size`を自動推定：

```python
# patch_level = log2(level0_magnification / target_magnification)
# 例:
#   40x scan → 20x target: log2(40/20) = 1
#   20x scan → 20x target: log2(20/20) = 0

# patch_sizeはh5属性から直接取得
patch_size = attrs.get('patch_size', 512)
```

### 7.3 データ形式

| 形式 | データ構造 | 座標系 |
|-----|----------|-------|
| trident (h5) | `features` (N, 768), `coords` (N, 2) | level 0座標 |
| SMMILe (npy) | `feature`, `index` | ターゲット倍率座標 |

### 7.4 ファイル命名規則

| ソース | ファイル名 |
|-------|----------|
| trident | `{slide_id}.h5` |
| SMMILe | `{slide_id}_{level}_{patch_size}.h5` |

---

## 8. 開発者向け情報

### 8.1 テストコード

`tests/test_trident_integration.py` で統合テストを実行できます。

```bash
# 全テスト実行
python tests/test_trident_integration.py --test_all

# 実際のh5ファイルでテスト
python tests/test_trident_integration.py --h5_path <path_to_your_h5>

# 個別テスト
python tests/test_trident_integration.py --test_coords
python tests/test_trident_integration.py --test_mixed
python tests/test_trident_integration.py --test_symlink
python tests/test_trident_integration.py --test_superpixel
python tests/test_trident_integration.py --test_dataset
```

### 8.2 テスト項目

1. **座標変換テスト**: 40x/20xスキャンでの座標変換が正しく行われるか
2. **混在倍率テスト**: 40xと20xが混在しても統一された座標系になるか
3. **symlink作成テスト**: 自動推定を含むsymlink作成が正しく動作するか
4. **superpixel生成テスト**: h5ファイルからsuperpixelが正しく生成されるか
5. **データセット読み込みテスト**: h5形式のデータが正しく読み込めるか

### 8.3 実装チェックリスト

#### Phase 1: symlinkスクリプト ✅
- [x] `create_smmile_structure.py`の作成
- [x] 自動推定機能の実装
- [x] ドライランモード
- [x] エラーハンドリング

#### Phase 2: superpixel_generation.py修正 ✅
- [x] `load_features_and_coords`関数の実装
- [x] 座標変換の実装
- [x] デバッグモードの追加

#### Phase 3: データセット読み込み修正 ✅
- [x] `single/datasets/dataset_nic.py`のh5対応
- [x] `multi/datasets/dataset_nic.py`のh5対応
- [x] 座標変換の実装

#### Phase 4: テストコード ✅
- [x] 座標変換テスト
- [x] 混在倍率テスト
- [x] symlink作成テスト
- [x] superpixel生成テスト
- [x] データセット読み込みテスト

### 8.4 修正されたファイルの詳細

| ファイル | 状態 | 主な変更内容 |
|---------|------|------------|
| `create_smmile_structure.py` | 新規作成 | symlink作成、自動推定 |
| `superpixel_generation.py` | 修正 | h5対応、座標変換 |
| `single/datasets/dataset_nic.py` | 修正 | h5対応、座標変換 |
| `multi/datasets/dataset_nic.py` | 修正 | h5対応、座標変換 |
| `tests/test_trident_integration.py` | 新規作成 | 統合テスト |
| `single/main.py` | 修正 | csv_path, label_dict対応 |
| `single/configs_custom/config_custom_template.yaml` | 新規作成 | テンプレート |
| `single/dataset_csv/sample_custom.csv` | 新規作成 | CSVサンプル |
| `generate_heatmap.py` | 修正 | 各クラス確率対応、coord_scale追加 |
| `single/eval.py` | 修正 | カスタムタスク対応 |
| `single/utils/eval_utils.py` | 修正 | 各クラス確率出力、ラベルなしデータ対応 |
