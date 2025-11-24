#!/usr/bin/env python
"""
Trident出力とSMMILe統合のテストスクリプト

テスト項目:
1. 座標変換の正確性（40x/20xスキャンでの変換）
2. h5ファイルの読み込み
3. symlink作成
4. superpixel生成
5. データセット読み込み

使用方法:
    python tests/test_trident_integration.py --h5_path <path_to_h5_file>
    python tests/test_trident_integration.py --test_all  # 全テストを実行（モックデータ使用）
"""

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np
import h5py

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_h5_file(output_path, level0_mag=40, target_mag=20, patch_size=512, num_patches=100, feature_dim=768):
    """
    テスト用のモックh5ファイルを作成
    
    Args:
        output_path: 出力先パス
        level0_mag: WSIのスキャン倍率
        target_mag: ターゲット倍率
        patch_size: ターゲット倍率でのパッチサイズ
        num_patches: パッチ数
        feature_dim: 特徴量次元
    """
    patch_size_level0 = patch_size * level0_mag // target_mag
    
    # level 0座標を生成（グリッド状）
    grid_size = int(np.ceil(np.sqrt(num_patches)))
    coords_level0 = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(coords_level0) >= num_patches:
                break
            coords_level0.append([i * patch_size_level0, j * patch_size_level0])
        if len(coords_level0) >= num_patches:
            break
    coords_level0 = np.array(coords_level0[:num_patches], dtype=np.int64)
    
    # ランダムな特徴量を生成
    features = np.random.randn(num_patches, feature_dim).astype(np.float32)
    
    # h5ファイルを保存
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=features)
        coords_ds = f.create_dataset('coords', data=coords_level0)
        
        # 属性を設定
        coords_ds.attrs['patch_size'] = patch_size
        coords_ds.attrs['patch_size_level0'] = patch_size_level0
        coords_ds.attrs['level0_magnification'] = level0_mag
        coords_ds.attrs['target_magnification'] = target_mag
        coords_ds.attrs['overlap'] = 0
        coords_ds.attrs['name'] = os.path.splitext(os.path.basename(output_path))[0]
    
    return coords_level0, features


def test_coordinate_conversion(h5_path=None, verbose=True):
    """
    座標変換のテスト
    
    h5ファイルから座標を読み込み、ターゲット倍率座標に変換する際に
    座標間隔がpatch_sizeと一致することを確認
    """
    print("\n" + "="*60)
    print("TEST: Coordinate Conversion")
    print("="*60)
    
    temp_dir = None
    
    try:
        if h5_path is None:
            # モックデータを作成
            temp_dir = tempfile.mkdtemp()
            h5_path = os.path.join(temp_dir, "test_slide.h5")
            
            # 40xスキャン、20xターゲットのテストケース
            coords_level0, _ = create_mock_h5_file(
                h5_path, level0_mag=40, target_mag=20, patch_size=512, num_patches=100
            )
            print(f"Created mock h5 file: {h5_path}")
        
        # h5ファイルを読み込み
        with h5py.File(h5_path, 'r') as f:
            coords_level0 = f['coords'][:]
            attrs = dict(f['coords'].attrs)
            
            patch_size = attrs.get('patch_size', 512)
            patch_size_level0 = attrs.get('patch_size_level0', patch_size)
            level0_mag = attrs.get('level0_magnification', 40)
            target_mag = attrs.get('target_magnification', 20)
        
        print(f"\nFile: {h5_path}")
        print(f"Attributes:")
        print(f"  level0_magnification: {level0_mag}")
        print(f"  target_magnification: {target_mag}")
        print(f"  patch_size: {patch_size}")
        print(f"  patch_size_level0: {patch_size_level0}")
        
        # 座標変換を実行
        scale_factor = target_mag / level0_mag
        coords_target = (coords_level0 * scale_factor).astype(np.int64)
        
        # 座標間隔を計算
        x_sorted = np.sort(np.unique(coords_level0[:, 0]))
        x_step_level0 = x_sorted[1] - x_sorted[0] if len(x_sorted) > 1 else 0
        
        x_sorted_target = np.sort(np.unique(coords_target[:, 0]))
        x_step_target = x_sorted_target[1] - x_sorted_target[0] if len(x_sorted_target) > 1 else 0
        
        print(f"\nCoordinate Analysis:")
        print(f"  Level 0 coords range: x=[{coords_level0[:,0].min()}, {coords_level0[:,0].max()}], y=[{coords_level0[:,1].min()}, {coords_level0[:,1].max()}]")
        print(f"  Level 0 coord step: {x_step_level0}")
        print(f"  Target coords range: x=[{coords_target[:,0].min()}, {coords_target[:,0].max()}], y=[{coords_target[:,1].min()}, {coords_target[:,1].max()}]")
        print(f"  Target coord step: {x_step_target}")
        
        # 検証
        expected_step_level0 = patch_size_level0
        expected_step_target = patch_size
        
        test_passed = True
        
        if x_step_level0 != expected_step_level0:
            print(f"\n[WARNING] Level 0 step mismatch: expected {expected_step_level0}, got {x_step_level0}")
            # これは警告のみ（元のファイルが期待通りでない可能性）
        
        if x_step_target != expected_step_target:
            print(f"\n[FAIL] Target step mismatch: expected {expected_step_target}, got {x_step_target}")
            test_passed = False
        else:
            print(f"\n[PASS] Target coord step matches patch_size: {x_step_target} == {expected_step_target}")
        
        return test_passed
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_mixed_magnification(verbose=True):
    """
    40xと20xスキャンが混在した場合のテスト
    両方とも同じターゲット座標系（patch_size=512）になることを確認
    """
    print("\n" + "="*60)
    print("TEST: Mixed Magnification (40x and 20x scans)")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 40xスキャンのファイル
        h5_40x = os.path.join(temp_dir, "slide_40x.h5")
        coords_40x_level0, _ = create_mock_h5_file(
            h5_40x, level0_mag=40, target_mag=20, patch_size=512, num_patches=50
        )
        
        # 20xスキャンのファイル
        h5_20x = os.path.join(temp_dir, "slide_20x.h5")
        coords_20x_level0, _ = create_mock_h5_file(
            h5_20x, level0_mag=20, target_mag=20, patch_size=512, num_patches=50
        )
        
        # 座標変換を実行
        # 40x -> 20x: scale_factor = 0.5
        coords_40x_target = (coords_40x_level0 * 0.5).astype(np.int64)
        
        # 20x -> 20x: scale_factor = 1.0
        coords_20x_target = (coords_20x_level0 * 1.0).astype(np.int64)
        
        # 座標間隔を確認
        x_sorted_40x = np.sort(np.unique(coords_40x_target[:, 0]))
        x_step_40x = x_sorted_40x[1] - x_sorted_40x[0] if len(x_sorted_40x) > 1 else 0
        
        x_sorted_20x = np.sort(np.unique(coords_20x_target[:, 0]))
        x_step_20x = x_sorted_20x[1] - x_sorted_20x[0] if len(x_sorted_20x) > 1 else 0
        
        print(f"\n40x scan (level0_mag=40, target_mag=20):")
        print(f"  Level 0 coord step: {np.sort(np.unique(coords_40x_level0[:,0]))[1] - np.sort(np.unique(coords_40x_level0[:,0]))[0]}")
        print(f"  Target coord step: {x_step_40x}")
        
        print(f"\n20x scan (level0_mag=20, target_mag=20):")
        print(f"  Level 0 coord step: {np.sort(np.unique(coords_20x_level0[:,0]))[1] - np.sort(np.unique(coords_20x_level0[:,0]))[0]}")
        print(f"  Target coord step: {x_step_20x}")
        
        # 検証
        test_passed = True
        
        if x_step_40x != 512:
            print(f"\n[FAIL] 40x scan: expected step 512, got {x_step_40x}")
            test_passed = False
        
        if x_step_20x != 512:
            print(f"\n[FAIL] 20x scan: expected step 512, got {x_step_20x}")
            test_passed = False
        
        if x_step_40x == x_step_20x == 512:
            print(f"\n[PASS] Both 40x and 20x scans have consistent target coord step: 512")
        
        return test_passed
        
    finally:
        shutil.rmtree(temp_dir)


def test_symlink_creation(verbose=True):
    """
    symlink作成機能のテスト（自動推定を含む）
    """
    print("\n" + "="*60)
    print("TEST: Symlink Creation (with auto-estimation)")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # テスト用のh5ファイルを作成（40xスキャンと20xスキャン）
        src_dir = os.path.join(temp_dir, "trident_features")
        os.makedirs(src_dir)
        
        # 40xスキャン → level=1
        h5_40x = os.path.join(src_dir, "slide_40x.h5")
        create_mock_h5_file(h5_40x, level0_mag=40, target_mag=20, patch_size=512)
        
        # 20xスキャン → level=0
        h5_20x = os.path.join(src_dir, "slide_20x.h5")
        create_mock_h5_file(h5_20x, level0_mag=20, target_mag=20, patch_size=512)
        
        # 出力ディレクトリ
        dst_dir = os.path.join(temp_dir, "smmile_features")
        
        # create_smmile_structure.pyをインポート
        from create_smmile_structure import create_relative_symlink, validate_h5_file, estimate_patch_level_from_h5
        
        # 自動推定のテスト
        print("\nTesting auto-estimation:")
        
        level_40x, size_40x = estimate_patch_level_from_h5(h5_40x)
        print(f"  40x scan: level={level_40x}, size={size_40x}")
        
        level_20x, size_20x = estimate_patch_level_from_h5(h5_20x)
        print(f"  20x scan: level={level_20x}, size={size_20x}")
        
        test_passed = True
        
        # 40xスキャンはlevel=1になるべき
        if level_40x != 1:
            print(f"\n[FAIL] 40x scan: expected level=1, got level={level_40x}")
            test_passed = False
        else:
            print(f"\n[PASS] 40x scan auto-estimated correctly: level={level_40x}")
        
        # 20xスキャンはlevel=0になるべき
        if level_20x != 0:
            print(f"[FAIL] 20x scan: expected level=0, got level={level_20x}")
            test_passed = False
        else:
            print(f"[PASS] 20x scan auto-estimated correctly: level={level_20x}")
        
        # symlinkの作成テスト
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_40x = os.path.join(dst_dir, f"slide_40x_{level_40x}_{size_40x}.h5")
        dst_20x = os.path.join(dst_dir, f"slide_20x_{level_20x}_{size_20x}.h5")
        
        create_relative_symlink(h5_40x, dst_40x)
        create_relative_symlink(h5_20x, dst_20x)
        
        if os.path.islink(dst_40x) and os.path.islink(dst_20x):
            print(f"[PASS] Symlinks created with auto-estimated names")
            print(f"  40x: {os.path.basename(dst_40x)}")
            print(f"  20x: {os.path.basename(dst_20x)}")
        else:
            print(f"[FAIL] Symlink creation failed")
            test_passed = False
        
        return test_passed
        
    finally:
        shutil.rmtree(temp_dir)


def test_superpixel_generation(verbose=True):
    """
    superpixel生成機能のテスト
    """
    print("\n" + "="*60)
    print("TEST: Superpixel Generation")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # テスト用のh5ファイルを作成
        h5_path = os.path.join(temp_dir, "test_slide_0_512.h5")
        create_mock_h5_file(h5_path, num_patches=200, feature_dim=768)
        
        # superpixel_generation.pyの関数をインポート
        from superpixel_generation import load_features_and_coords, get_nic_with_coord, generate_adjacency_matrix
        from skimage import segmentation
        
        # 特徴量と座標を読み込み
        features, coords_nd, inst_label, actual_patch_size = load_features_and_coords(
            h5_path, target_patch_size=512
        )
        
        print(f"\nLoaded data:")
        print(f"  Features shape: {features.shape}")
        print(f"  Coords shape: {coords_nd.shape}")
        print(f"  Patch size: {actual_patch_size}")
        
        # 座標間隔を確認
        x_sorted = np.sort(np.unique(coords_nd[:, 0]))
        x_step = x_sorted[1] - x_sorted[0] if len(x_sorted) > 1 else 0
        print(f"  Coord step: {x_step}")
        
        # NIC変換
        if inst_label is None:
            inst_label = [0 for _ in range(coords_nd.shape[0])]
        
        features_norm = (features - features.min()) / (features.max() - features.min()) * 255
        features_nic, mask, _ = get_nic_with_coord(features_norm, coords_nd, actual_patch_size, inst_label)
        
        print(f"\nNIC conversion:")
        print(f"  Features NIC shape: {features_nic.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Non-zero mask count: {np.sum(mask)}")
        
        # SLIC superpixel
        data = np.transpose(features_nic, (1, 2, 0))
        n_segments = max(10, int(features.shape[0] / 16))
        m_slic = segmentation.slic(data, n_segments=n_segments, mask=mask, compactness=50, start_label=1).T
        
        # 隣接行列
        m_adj = generate_adjacency_matrix(m_slic)
        
        print(f"\nSuperpixel generation:")
        print(f"  m_slic shape: {m_slic.shape}")
        print(f"  m_slic unique values: {len(np.unique(m_slic))}")
        print(f"  m_adj shape: {m_adj.shape}")
        
        # 検証
        test_passed = True
        
        if x_step != actual_patch_size:
            print(f"\n[FAIL] Coord step mismatch: expected {actual_patch_size}, got {x_step}")
            test_passed = False
        else:
            print(f"\n[PASS] Coord step matches patch_size: {x_step}")
        
        if np.max(m_slic) != (m_adj.shape[0] - 1):
            print(f"\n[FAIL] Superpixel count mismatch: max(m_slic)={np.max(m_slic)}, m_adj.shape[0]-1={m_adj.shape[0]-1}")
            test_passed = False
        else:
            print(f"[PASS] Superpixel count consistent: {np.max(m_slic) + 1} superpixels")
        
        return test_passed
        
    finally:
        shutil.rmtree(temp_dir)


def load_features_and_coords_h5_standalone(h5_path):
    """
    h5ファイル（trident形式）から特徴量と座標を読み込む（スタンドアロン版）
    dataset_nic.pyと同じロジックだが、依存関係なしで動作
    """
    import torch
    
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:])
        coords_level0 = f['coords'][:]
        
        # 属性から倍率情報を取得
        if 'coords' in f and hasattr(f['coords'], 'attrs'):
            attrs = dict(f['coords'].attrs)
            patch_size = attrs.get('patch_size', 512)
            level0_mag = attrs.get('level0_magnification', 40)
            target_mag = attrs.get('target_magnification', 20)
        else:
            patch_size = 512
            level0_mag = 20
            target_mag = 20
        
        # 座標をターゲット倍率座標に変換
        scale_factor = target_mag / level0_mag
        coords_nd = (coords_level0 * scale_factor).astype(np.int64)
    
    # h5形式にはinst_labelがないので、全て-1（アノテーションなし）を設定
    inst_label = [-1 for _ in range(coords_nd.shape[0])]
    
    return features, coords_nd, inst_label, patch_size


def test_dataset_loading(verbose=True):
    """
    データセット読み込み機能のテスト
    """
    print("\n" + "="*60)
    print("TEST: Dataset Loading (h5 format)")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # テスト用のディレクトリ構造を作成
        feature_dir = os.path.join(temp_dir, "features")
        sp_dir = os.path.join(temp_dir, "superpixels")
        os.makedirs(feature_dir)
        os.makedirs(sp_dir)
        
        # テスト用のh5ファイルを作成
        slide_id = "test_slide"
        data_mag = "0_512"
        h5_path = os.path.join(feature_dir, f"{slide_id}_{data_mag}.h5")
        create_mock_h5_file(h5_path, num_patches=100, feature_dim=512)
        
        # テスト用のsuperpixelファイルを作成
        from superpixel_generation import load_features_and_coords, get_nic_with_coord, generate_adjacency_matrix
        from skimage import segmentation
        
        features, coords_nd, inst_label, patch_size = load_features_and_coords(h5_path, target_patch_size=512)
        if inst_label is None:
            inst_label = [0 for _ in range(coords_nd.shape[0])]
        
        features_norm = (features - features.min()) / (features.max() - features.min()) * 255
        features_nic, mask, _ = get_nic_with_coord(features_norm, coords_nd, patch_size, inst_label)
        
        data = np.transpose(features_nic, (1, 2, 0))
        n_segments = max(10, int(features.shape[0] / 16))
        m_slic = segmentation.slic(data, n_segments=n_segments, mask=mask, compactness=50, start_label=1).T
        m_adj = generate_adjacency_matrix(m_slic)
        
        sp_path = os.path.join(sp_dir, f"{slide_id}_{data_mag}.npy")
        np.save(sp_path, {'m_slic': m_slic, 'm_adj': m_adj})
        
        print(f"\nCreated test files:")
        print(f"  Feature file: {h5_path}")
        print(f"  Superpixel file: {sp_path}")
        
        # h5読み込みテスト（スタンドアロン関数を使用）
        features_loaded, coords_loaded, inst_label_loaded, patch_size_loaded = load_features_and_coords_h5_standalone(h5_path)
        
        print(f"\nLoaded via load_features_and_coords_h5_standalone:")
        print(f"  Features shape: {features_loaded.shape}")
        print(f"  Coords shape: {coords_loaded.shape}")
        print(f"  Patch size: {patch_size_loaded}")
        
        # 座標間隔を確認
        x_sorted = np.sort(np.unique(coords_loaded[:, 0]))
        x_step = x_sorted[1] - x_sorted[0] if len(x_sorted) > 1 else 0
        print(f"  Coord step: {x_step}")
        
        # 検証
        test_passed = True
        
        if x_step != patch_size_loaded:
            print(f"\n[FAIL] Coord step mismatch: expected {patch_size_loaded}, got {x_step}")
            test_passed = False
        else:
            print(f"\n[PASS] Coord step matches patch_size: {x_step}")
        
        return test_passed
        
    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """全テストを実行"""
    print("\n" + "#"*60)
    print("# Running All Tests")
    print("#"*60)
    
    results = {}
    
    results['coordinate_conversion'] = test_coordinate_conversion()
    results['mixed_magnification'] = test_mixed_magnification()
    results['symlink_creation'] = test_symlink_creation()
    results['superpixel_generation'] = test_superpixel_generation()
    results['dataset_loading'] = test_dataset_loading()
    
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*60)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test Trident-SMMILe integration")
    parser.add_argument('--h5_path', type=str, default=None,
                        help='Path to a real h5 file for testing')
    parser.add_argument('--test_all', action='store_true',
                        help='Run all tests with mock data')
    parser.add_argument('--test_coords', action='store_true',
                        help='Test coordinate conversion only')
    parser.add_argument('--test_mixed', action='store_true',
                        help='Test mixed magnification only')
    parser.add_argument('--test_symlink', action='store_true',
                        help='Test symlink creation only')
    parser.add_argument('--test_superpixel', action='store_true',
                        help='Test superpixel generation only')
    parser.add_argument('--test_dataset', action='store_true',
                        help='Test dataset loading only')
    
    args = parser.parse_args()
    
    if args.test_all:
        success = run_all_tests()
        return 0 if success else 1
    
    if args.h5_path:
        test_coordinate_conversion(args.h5_path)
        return 0
    
    if args.test_coords:
        test_coordinate_conversion()
    elif args.test_mixed:
        test_mixed_magnification()
    elif args.test_symlink:
        test_symlink_creation()
    elif args.test_superpixel:
        test_superpixel_generation()
    elif args.test_dataset:
        test_dataset_loading()
    else:
        # デフォルトは全テスト
        run_all_tests()
    
    return 0


if __name__ == "__main__":
    exit(main())

