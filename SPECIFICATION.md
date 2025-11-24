# SMMILe Fork: Trident出力対応のための詳細仕様書

## 概要
tridentで抽出したCONCHv1.5特徴量（h5形式）をSMMILeで使用できるようにする。以下のコンポーネントを実装済み：
1. symlinkベースのディレクトリ構造変換スクリプト（自動推定機能付き）
2. SMMILeコードのh5対応修正（座標変換込み）
3. テストコード

---

## 実装済みファイル一覧

| ファイル | 状態 | 説明 |
|---------|------|------|
| `create_smmile_structure.py` | 新規作成 | symlink作成スクリプト |
| `superpixel_generation.py` | 修正 | h5対応・座標変換追加 |
| `single/datasets/dataset_nic.py` | 修正 | h5対応・座標変換追加 |
| `multi/datasets/dataset_nic.py` | 修正 | h5対応・座標変換追加 |
| `tests/test_trident_integration.py` | 新規作成 | 統合テスト |

---

## 1. Symlinkスクリプト仕様 (`create_smmile_structure.py`)

### 1.1 目的
tridentの出力ディレクトリからSMMILeが期待するディレクトリ構造をsymlinkで作成する。

### 1.2 入力パラメータ

```python
--trident_features_dir: str (必須)
    tridentの特徴量ディレクトリパス
    例: "trident_processed/20x_512px_0px_overlap/features_conch_v15"
    
--output_dir: str (必須)
    SMMILe用の出力ディレクトリパス
    例: "smmile_data/features_conch_v15"
    
--patch_level: int, default=None
    パッチレベル（指定しない場合はh5から自動推定）
    
--patch_size: int, default=None
    パッチサイズ（指定しない場合はh5から自動推定）
    
--no_auto_estimate: bool, default=False
    自動推定を無効化（--patch_levelと--patch_sizeが必須になる）
    
--slide_list: str, optional
    処理対象スライドのリストファイル（csvまたはtxt）
    指定しない場合は全h5ファイルを処理
    
--file_pattern: str, default="*.h5"
    処理するファイルのパターン
    
--dry_run: bool, default=False
    実際にsymlinkを作成せず、処理内容のみ表示
    
--skip_validation: bool, default=False
    h5ファイルの検証をスキップ（高速化）
    
--verbose: bool, default=False
    詳細な出力を表示
```

### 1.3 自動推定機能

h5ファイルの属性から`patch_level`と`patch_size`を自動推定：

```python
# patch_level = log2(level0_magnification / target_magnification)
# 例:
#   40x scan → 20x target: log2(40/20) = 1
#   20x scan → 20x target: log2(20/20) = 0
#   40x scan → 10x target: log2(40/10) = 2

# patch_sizeはh5属性から直接取得
patch_size = attrs.get('patch_size', 512)
```

### 1.4 処理フロー

```
1. 入力検証
   - trident_features_dirの存在確認
   - h5ファイルの一覧取得
   - slide_listが指定されている場合、リスト読み込み

2. 出力ディレクトリ作成
   - output_dir/の作成

3. 各h5ファイルに対して
   a. 自動推定（デフォルト）
      - h5属性からpatch_levelとpatch_sizeを推定
      
   b. 新ファイル名生成
      出力: "{slide_id}_{patch_level}_{patch_size}.h5"
      例: "TCGA-XXX_1_512.h5" (40xスキャン)
          "TCGA-YYY_0_512.h5" (20xスキャン)
      
   c. h5ファイル内容の検証（skip_validationでスキップ可能）
      - 'features'キーの存在確認
      - 'coords'キーの存在確認
      - shapeの整合性確認
      
   d. symlinkの作成
      相対パスでsymlinkを作成

4. 処理結果のサマリー出力
   - 処理成功数
   - スキップ数とその理由
   - エラー数
   - patch_level/patch_size分布
```

### 1.5 出力ディレクトリ構造

```
output_dir/
├── TCGA-XXX_1_512.h5 -> ../../trident_processed/.../TCGA-XXX.h5  (40xスキャン)
├── TCGA-YYY_0_512.h5 -> ../../trident_processed/.../TCGA-YYY.h5  (20xスキャン)
└── ...
```

### 1.6 使用例

```bash
# 基本的な使用（自動推定、推奨）
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15

# ドライラン（実際には作成しない、確認用）
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15 \
    --dry_run --verbose

# 自動推定を無効化して固定値を使用
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15 \
    --no_auto_estimate \
    --patch_level 0 \
    --patch_size 512

# スライドリストを指定
python create_smmile_structure.py \
    --trident_features_dir trident_processed/20x_512px_0px_overlap/features_conch_v15 \
    --output_dir smmile_data/features_conch_v15 \
    --slide_list slide_list.csv
```

---

## 2. 座標変換の仕様

### 2.1 問題点

tridentの座標はWSI level 0座標で保存されている。スキャン倍率によって座標間隔が異なる：

| スキャン倍率 | ターゲット倍率 | patch_size | patch_size_level0 | 座標間隔 |
|------------|--------------|------------|-------------------|----------|
| 40x | 20x | 512 | 1024 | 1024 |
| 20x | 20x | 512 | 512 | 512 |

### 2.2 解決策

h5ファイルの属性から倍率情報を取得し、座標をターゲット倍率座標に変換：

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

### 2.3 変換結果

変換後は、スキャン倍率に関わらず座標間隔が`patch_size`に統一される：

```
40x scan: [4096, 19456] → [2048, 9728] (step: 1024 → 512)
20x scan: [2048, 9728]  → [2048, 9728] (step: 512 → 512)
```

---

## 3. superpixel_generation.py修正仕様

### 3.1 修正内容

- h5形式（trident出力）の読み込みを追加
- 座標変換を実装
- ファイル拡張子で自動判定して処理を分岐
- 既存のnpy形式のサポートを維持

### 3.2 `load_features_and_coords`関数

```python
def load_features_and_coords(fea_path, keyword_feature='feature', target_patch_size=None):
    """
    特徴量と座標を読み込む（h5とnpyの両方に対応）
    
    h5形式の場合、座標はWSI level 0座標で保存されているため、
    ターゲット倍率座標に変換します。
    
    Args:
        fea_path: 特徴量ファイルのパス（.h5 or .npy）
        keyword_feature: npy形式の場合の特徴量キー名
        target_patch_size: h5形式の場合、座標変換後のパッチサイズ
        
    Returns:
        features: (N, D) ndarray
        coords_nd: (N, 2) ndarray - ターゲット倍率での座標
        inst_label: list or None
        actual_patch_size: 座標系でのパッチサイズ
    """
```

### 3.3 使用例

```bash
# h5ファイル（trident出力）を処理
python superpixel_generation.py \
    --fea_dir smmile_data/features_conch_v15 \
    --sp_dir smmile_data/superpixels_n16_c50_512 \
    --file_suffix '*_0_512.h5' \
    --n_segments_persp 16 \
    --compactness 50

# 40xスキャンと20xスキャンが混在する場合
python superpixel_generation.py \
    --fea_dir smmile_data/features_conch_v15 \
    --file_suffix '*.h5' \
    --n_segments_persp 16 \
    --compactness 50

# npyも引き続きサポート
python superpixel_generation.py \
    --fea_dir path/to/npy/features \
    --file_suffix '*_1_512.npy' \
    --keyword_feature feature \
    --n_segments_persp 16 \
    --compactness 50
```

---

## 4. データセット読み込みの修正

### 4.1 修正ファイル

- `single/datasets/dataset_nic.py`
- `multi/datasets/dataset_nic.py`

### 4.2 主な変更点

1. h5ファイルの読み込み対応
2. 座標変換（level 0 → ターゲット倍率）
3. inst_labelがない場合の処理（全て-1を設定）
4. ファイル拡張子で自動判定（h5優先、npyフォールバック）

### 4.3 ファイル検索ロジック

```python
def _get_feature_path(self, slide_id):
    """
    特徴量ファイルのパスを取得（h5とnpyの両方をサポート）
    """
    # まずh5ファイルを探す
    h5_path = os.path.join(self.data_dir, '{}_{}.h5'.format(slide_id, self.data_mag))
    if os.path.exists(h5_path):
        return h5_path
    
    # h5がなければnpyファイルを探す
    npy_path = os.path.join(self.data_dir, '{}_{}.npy'.format(slide_id, self.data_mag))
    return npy_path
```

---

## 5. テストコード

### 5.1 テストファイル

`tests/test_trident_integration.py`

### 5.2 テスト項目

1. **座標変換テスト**: 40x/20xスキャンでの座標変換が正しく行われるか
2. **混在倍率テスト**: 40xと20xが混在しても統一された座標系になるか
3. **symlink作成テスト**: 自動推定を含むsymlink作成が正しく動作するか
4. **superpixel生成テスト**: h5ファイルからsuperpixelが正しく生成されるか
5. **データセット読み込みテスト**: h5形式のデータが正しく読み込めるか

### 5.3 使用方法

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

---

## 6. configの設定例

### 6.1 h5形式（trident出力）を使用する場合

```yaml
# configs/config_example_h5.yaml
n_classes: 2
task: 'your_task'

# パス設定
data_root_dir: 'smmile_data/features_conch_v15'
data_sp_dir: 'smmile_data/superpixels_n16_c50_512'

# data_magはファイル名用の識別子
# 40xスキャンなら '1_512'、20xスキャンなら '0_512'
# 混在する場合はCSVで各スライドのdata_magを管理するか、
# 統一されたdata_magを使用（symlinkで対応）
data_mag: '0_512'

# patch_sizeはh5から自動取得されるが、npyフォールバック用に設定
patch_size: 512

# CONCHv1.5の特徴量次元
fea_dim: 768

# その他の設定...
```

### 6.2 混在スキャン倍率の対応

40xと20xスキャンが混在する場合、以下の方法で対応：

1. **symlinkで統一**（推奨）: `create_smmile_structure.py`で自動推定を使用
2. **CSVでスライドごとに管理**: slide_idごとに適切なdata_magを設定

---

## 7. エラー対処ガイド

### 7.1 よくあるエラー

**エラー1**: `KeyError: 'features'` または `KeyError: 'coords'`
- **原因**: h5ファイルの構造が期待と異なる
- **対処**: h5ファイルの内容を確認
```python
import h5py
with h5py.File('file.h5', 'r') as f:
    print(list(f.keys()))
```

**エラー2**: 座標間隔がpatch_sizeと一致しない
- **原因**: 座標変換が正しく行われていない
- **対処**: h5属性を確認
```python
import h5py
with h5py.File('file.h5', 'r') as f:
    print(dict(f['coords'].attrs))
```

**エラー3**: symlink作成失敗
- **原因**: ファイルシステムがsymlinkをサポートしていない
- **対処**: 絶対パスでsymlinkを作成するか、ファイルをコピー

### 7.2 デバッグモード

```bash
# superpixel_generation.pyのデバッグ出力
python superpixel_generation.py \
    --fea_dir <DIR> \
    --debug

# テストでの詳細確認
python tests/test_trident_integration.py --h5_path <FILE>
```

---

## 8. 実装チェックリスト

### Phase 1: symlinkスクリプト ✅
- [x] `create_smmile_structure.py`の作成
- [x] argparseでパラメータ定義
- [x] h5ファイルの検証関数実装
- [x] 相対パスsymlink作成関数実装
- [x] **自動推定機能の実装（patch_level, patch_size）**
- [x] ドライランモードの実装
- [x] エラーハンドリング
- [x] ロギング・サマリー出力

### Phase 2: superpixel_generation.py修正 ✅
- [x] `load_features_and_coords`関数の実装
- [x] **座標変換の実装（level 0 → ターゲット倍率）**
- [x] `process_file`関数の修正
- [x] argparseの更新
- [x] エラーハンドリングの追加
- [x] デバッグモードの追加

### Phase 3: データセット読み込み修正 ✅
- [x] `single/datasets/dataset_nic.py`のh5対応
- [x] `multi/datasets/dataset_nic.py`のh5対応
- [x] **座標変換の実装**
- [x] inst_label処理の追加

### Phase 4: テストコード ✅
- [x] 座標変換テスト
- [x] 混在倍率テスト
- [x] symlink作成テスト（自動推定含む）
- [x] superpixel生成テスト
- [x] データセット読み込みテスト

---

## 9. 重要な注意事項

### 9.1 データ形式について
- trident出力: h5形式で`features` (N, 768) と `coords` (N, 2) - **level 0座標**
- SMMILe元々の形式: npy形式で`feature`と`index`
- **座標変換により、どちらの形式もターゲット倍率座標で統一される**

### 9.2 ファイル命名規則
- trident: `{slide_id}.h5`
- SMMILe期待: `{slide_id}_{level}_{patch_size}.h5`
- **symlinkで変換、levelは自動推定**

### 9.3 座標系の注意
- 両形式とも(x, y)の順序で統一
- **tridentの座標はlevel 0座標で保存されているため、読み込み時に変換が必要**
- 変換後はスキャン倍率に関わらず座標間隔がpatch_sizeに統一される

この仕様書に基づいて実装されています。
