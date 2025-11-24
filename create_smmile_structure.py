#!/usr/bin/env python
"""
Trident出力からSMMILe用のディレクトリ構造をsymlinkで作成するスクリプト

Tridentの特徴量ファイル（h5形式）をSMMILeが期待する命名規則に変換します。
ファイル名: {slide_id}.h5 -> {slide_id}_{patch_level}_{patch_size}.h5
"""

import os
import argparse
import glob
import h5py
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Optional


def validate_h5_file(h5_path: str) -> Tuple[bool, str]:
    """
    h5ファイルが正しい形式か検証
    
    Args:
        h5_path: h5ファイルのパス
        
    Returns:
        (成功/失敗, エラーメッセージ)
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'features' not in f:
                return False, "Missing 'features' key"
            if 'coords' not in f:
                return False, "Missing 'coords' key"
            
            features = f['features']
            coords = f['coords']
            
            if len(features.shape) != 2:
                return False, f"Invalid features shape: {features.shape}"
            if len(coords.shape) != 2 or coords.shape[1] != 2:
                return False, f"Invalid coords shape: {coords.shape}"
            if features.shape[0] != coords.shape[0]:
                return False, f"Shape mismatch: features {features.shape[0]} vs coords {coords.shape[0]}"
                
        return True, ""
    except Exception as e:
        return False, str(e)


def get_h5_info(h5_path: str) -> dict:
    """
    h5ファイルの属性情報を取得
    
    Args:
        h5_path: h5ファイルのパス
        
    Returns:
        属性情報の辞書
    """
    info = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            info['num_patches'] = f['features'].shape[0]
            info['feature_dim'] = f['features'].shape[1]
            
            # coords属性から情報を取得
            if 'coords' in f and hasattr(f['coords'], 'attrs'):
                attrs = dict(f['coords'].attrs)
                info.update(attrs)
    except Exception as e:
        info['error'] = str(e)
    return info


def estimate_patch_level_from_h5(h5_path: str) -> Tuple[int, int]:
    """
    h5ファイルの属性からpatch_levelとpatch_sizeを推定
    
    patch_level = log2(level0_magnification / target_magnification)
    例:
      - 40x scan, 20x target → level = log2(40/20) = 1
      - 20x scan, 20x target → level = log2(20/20) = 0
      - 40x scan, 10x target → level = log2(40/10) = 2
    
    Args:
        h5_path: h5ファイルのパス
        
    Returns:
        (patch_level, patch_size) のタプル
    """
    import math
    
    with h5py.File(h5_path, 'r') as f:
        if 'coords' in f and hasattr(f['coords'], 'attrs'):
            attrs = dict(f['coords'].attrs)
            level0_mag = attrs.get('level0_magnification', 20)
            target_mag = attrs.get('target_magnification', 20)
            patch_size = attrs.get('patch_size', 512)
        else:
            # 属性がない場合はデフォルト値
            level0_mag = 20
            target_mag = 20
            patch_size = 512
    
    # patch_levelを計算
    if target_mag > 0 and level0_mag > 0:
        scale_factor = level0_mag / target_mag
        patch_level = int(round(math.log2(scale_factor)))
    else:
        patch_level = 0
    
    return patch_level, patch_size


def create_relative_symlink(src_path: str, dst_path: str) -> None:
    """
    相対パスでsymlinkを作成
    
    Args:
        src_path: ソースファイルのパス（絶対パスまたは相対パス）
        dst_path: デスティネーションファイルのパス（絶対パス）
    """
    src_abs = os.path.abspath(src_path)
    dst_abs = os.path.abspath(dst_path)
    dst_dir = os.path.dirname(dst_abs)
    
    # 相対パスを計算
    rel_path = os.path.relpath(src_abs, dst_dir)
    
    # 既存のsymlinkがある場合は削除
    if os.path.islink(dst_abs):
        os.unlink(dst_abs)
    elif os.path.exists(dst_abs):
        raise FileExistsError(f"File already exists (not a symlink): {dst_abs}")
    
    os.symlink(rel_path, dst_abs)


def load_slide_list(slide_list_path: str) -> List[str]:
    """
    スライドリストを読み込む
    
    Args:
        slide_list_path: CSVまたはTXTファイルのパス
        
    Returns:
        スライドIDのリスト
    """
    ext = os.path.splitext(slide_list_path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(slide_list_path)
        # slide_id列を探す
        for col in ['slide_id', 'slide', 'name', 'filename']:
            if col in df.columns:
                return df[col].astype(str).tolist()
        # 最初の列を使用
        return df.iloc[:, 0].astype(str).tolist()
    else:
        # TXTファイルとして読み込み
        with open(slide_list_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Trident出力からSMMILe用のディレクトリ構造をsymlinkで作成"
    )
    parser.add_argument(
        '--trident_features_dir', type=str, required=True,
        help='Tridentの特徴量ディレクトリパス (例: trident_processed/20x_512px_0px_overlap/features_conch_v15)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='SMMILe用の出力ディレクトリパス (例: smmile_data/features_conch_v15)'
    )
    parser.add_argument(
        '--patch_level', type=int, default=None,
        help='パッチレベル（ファイル名用の識別子）。指定しない場合はh5から自動推定'
    )
    parser.add_argument(
        '--patch_size', type=int, default=None,
        help='パッチサイズ（ファイル名用の識別子）。指定しない場合はh5から自動推定'
    )
    parser.add_argument(
        '--auto_estimate', action='store_true', default=True,
        help='h5ファイルからpatch_levelとpatch_sizeを自動推定（デフォルト: True）'
    )
    parser.add_argument(
        '--no_auto_estimate', action='store_true',
        help='自動推定を無効化し、--patch_levelと--patch_sizeを必須にする'
    )
    parser.add_argument(
        '--slide_list', type=str, default=None,
        help='処理対象スライドのリストファイル（CSVまたはTXT）'
    )
    parser.add_argument(
        '--file_pattern', type=str, default='*.h5',
        help='処理するファイルのパターン (default: *.h5)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='実際にsymlinkを作成せず、処理内容のみ表示'
    )
    parser.add_argument(
        '--skip_validation', action='store_true',
        help='h5ファイルの検証をスキップ（高速化）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='詳細な出力を表示'
    )
    
    args = parser.parse_args()
    
    # 自動推定の設定
    use_auto_estimate = not args.no_auto_estimate
    
    # 自動推定を無効化した場合、patch_levelとpatch_sizeが必須
    if not use_auto_estimate:
        if args.patch_level is None or args.patch_size is None:
            print("Error: --patch_level and --patch_size are required when --no_auto_estimate is set")
            return 1
    
    # 入力ディレクトリの確認
    if not os.path.isdir(args.trident_features_dir):
        print(f"Error: Trident features directory not found: {args.trident_features_dir}")
        return 1
    
    # h5ファイルの一覧取得
    h5_files = glob.glob(os.path.join(args.trident_features_dir, args.file_pattern))
    print(f"Found {len(h5_files)} h5 files in {args.trident_features_dir}")
    
    if use_auto_estimate:
        print("Auto-estimation enabled: patch_level and patch_size will be read from each h5 file")
    
    if len(h5_files) == 0:
        print("No h5 files found. Check the directory path and file pattern.")
        return 1
    
    # スライドリストでフィルタリング
    if args.slide_list:
        if not os.path.exists(args.slide_list):
            print(f"Error: Slide list file not found: {args.slide_list}")
            return 1
        
        target_slides = set(load_slide_list(args.slide_list))
        print(f"Filtering by slide list: {len(target_slides)} slides")
        
        filtered_files = []
        for h5_path in h5_files:
            slide_id = os.path.splitext(os.path.basename(h5_path))[0]
            if slide_id in target_slides:
                filtered_files.append(h5_path)
        
        h5_files = filtered_files
        print(f"After filtering: {len(h5_files)} h5 files")
    
    # 出力ディレクトリ作成
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 処理結果のカウンター
    success_count = 0
    skip_count = 0
    error_count = 0
    skip_reasons = {}
    
    # patch_level/patch_sizeの統計（自動推定の場合）
    level_size_stats = {}
    
    # 各h5ファイルを処理
    for h5_path in tqdm(h5_files, desc="Processing"):
        # ファイル名解析
        base_name = os.path.basename(h5_path)
        slide_id = os.path.splitext(base_name)[0]
        
        # patch_levelとpatch_sizeを決定
        if use_auto_estimate:
            try:
                patch_level, patch_size = estimate_patch_level_from_h5(h5_path)
            except Exception as e:
                if args.verbose:
                    print(f"Skip (cannot estimate patch_level): {slide_id} - {e}")
                skip_count += 1
                skip_reasons["Cannot estimate patch_level"] = skip_reasons.get("Cannot estimate patch_level", 0) + 1
                continue
        else:
            patch_level = args.patch_level
            patch_size = args.patch_size
        
        # 統計を記録
        key = f"{patch_level}_{patch_size}"
        level_size_stats[key] = level_size_stats.get(key, 0) + 1
        
        # 新ファイル名生成
        new_name = f"{slide_id}_{patch_level}_{patch_size}.h5"
        dst_path = os.path.join(args.output_dir, new_name)
        
        # h5ファイルの検証
        if not args.skip_validation:
            is_valid, error_msg = validate_h5_file(h5_path)
            if not is_valid:
                if args.verbose:
                    print(f"Skip (validation failed): {slide_id} - {error_msg}")
                skip_count += 1
                skip_reasons[error_msg] = skip_reasons.get(error_msg, 0) + 1
                continue
        
        # dry runモード
        if args.dry_run:
            if args.verbose:
                info = get_h5_info(h5_path)
                print(f"Would create: {dst_path}")
                print(f"  -> {h5_path}")
                print(f"  patch_level={patch_level}, patch_size={patch_size}")
                print(f"  Info: {info}")
            success_count += 1
            continue
        
        # symlinkの作成
        try:
            create_relative_symlink(h5_path, dst_path)
            success_count += 1
            if args.verbose:
                print(f"Created: {new_name} (level={patch_level}, size={patch_size})")
        except FileExistsError as e:
            if args.verbose:
                print(f"Skip (file exists): {slide_id}")
            skip_count += 1
            skip_reasons["File already exists"] = skip_reasons.get("File already exists", 0) + 1
        except Exception as e:
            print(f"Error creating symlink for {slide_id}: {e}")
            error_count += 1
    
    # サマリー出力
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total files processed: {len(h5_files)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    
    if skip_reasons:
        print("\nSkip reasons:")
        for reason, count in skip_reasons.items():
            print(f"  - {reason}: {count}")
    
    if level_size_stats:
        print("\nPatch level/size distribution:")
        for key, count in sorted(level_size_stats.items()):
            level, size = key.split('_')
            print(f"  - level={level}, size={size}: {count} files")
    
    if not args.dry_run:
        print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")
        if use_auto_estimate:
            print("File naming: {slide_id}_{auto_level}_{auto_size}.h5 (auto-estimated per file)")
        else:
            print(f"File naming: {{slide_id}}_{args.patch_level}_{args.patch_size}.h5")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())

