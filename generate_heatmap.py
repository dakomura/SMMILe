import os
import pandas as pd
import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from matplotlib import colormaps  # Import matplotlib colormap
from matplotlib.colors import Normalize
import argparse

# Initialize a colormap (e.g., 'viridis', 'plasma', 'coolwarm', 'hot')
colormap = colormaps.get_cmap('jet')  # Use the 'jet' colormap for a heatmap effect
norm = Normalize(vmin=0, vmax=1)  # Normalize prob values to the range [0, 1]

def create_prediction_overlay(svs_file, df_result, prob_column='prob', thumbnail_size=(1024, 1024), coord_scale=1.0):
    """
    SVSファイルに予測確率のオーバーレイを作成
    
    Args:
        svs_file: SVSファイルのパス
        df_result: 予測結果のDataFrame
        prob_column: 使用する確率の列名（'prob', 'prob_0', 'prob_1'など）
        thumbnail_size: サムネイルのサイズ
        coord_scale: 座標のスケーリングファクター（CSVの座標をLevel 0に変換するため）
    """
    # Step 1: Load the SVS file
    slide = openslide.OpenSlide(svs_file)
    
    # Step 2: Generate a thumbnail of the SVS file
    thumbnail = slide.get_thumbnail(thumbnail_size)
    thumbnail = thumbnail.convert("RGBA")  # Ensure RGBA for transparency support
    
    # Step 3: Prepare a blank mask image the same size as the thumbnail
    mask = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))  # Transparent initially
    
    # Step 4: Iterate through the DataFrame to extract patch info and add to the mask
    draw = ImageDraw.Draw(mask)
    
    # Calculate downscale factor based on thumbnail size vs full resolution
    downscale_factor = max(slide.level_dimensions[0][0], slide.level_dimensions[0][1]) / thumbnail_size[0]
    
    for _, row in df_result.iterrows():
        # Parse the x, y coordinates from the filename
        filename = row['filename']
        coords = filename.split('/')[-1].split('_')
        # Apply coordinate scaling to convert to Level 0 coordinates
        x, y = int(int(coords[0]) * coord_scale), int(int(coords[1]) * coord_scale)
        patch_size = int(int(coords[2].split('.')[0]) * coord_scale)
    
        # Downscale coordinates to match thumbnail size
        x_downscaled = int(x / downscale_factor)
        y_downscaled = int(y / downscale_factor)
        patch_size_downscaled = int(patch_size / downscale_factor)
    
        # Set color based on prediction probability
        prob_value = row[prob_column]
        if pd.notna(prob_value) and 0 <= prob_value <= 1:
            rgba = colormap(norm(prob_value))  # Map prob to colormap
            r, g, b, a = [int(c * 255) for c in rgba]  # Convert to 0-255 range
            color = (r, g, b, 150)
        else:
            color = (0, 0, 0, 0)  # Fully transparent for invalid prob values
    
        # Draw the rectangle on the mask
        draw.rectangle([x_downscaled, y_downscaled, x_downscaled + patch_size_downscaled, y_downscaled + patch_size_downscaled], fill=color)
    
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    combined = Image.alpha_composite(thumbnail, mask)
    return thumbnail, combined

def process_wsi(svs_file, df_results, output_dir, suffix, thumbnail_size, class_labels=None, coord_scale=1.0):
    """
    WSIを処理して各クラスのheatmapを生成
    
    Args:
        svs_file: SVSファイルのパス
        df_results: 予測結果のDataFrame
        output_dir: 出力ディレクトリ
        suffix: ファイル名のサフィックス
        thumbnail_size: サムネイルのサイズ
        class_labels: クラスラベルの辞書 {0: 'class0_name', 1: 'class1_name', ...}
        coord_scale: 座標のスケーリングファクター（CSVの座標をLevel 0に変換するため）
    """
    svs_name = os.path.splitext(os.path.basename(svs_file))[0]
    df_results_sub = df_results[df_results['svs_name'] == svs_name]
    
    if df_results_sub.empty:
        return None  # Skip if no results for this WSI
    
    # 各クラスの確率列を検出
    prob_columns = [col for col in df_results_sub.columns if col.startswith('prob_')]
    
    if not prob_columns:
        # 従来の単一prob列のみの場合
        prob_columns = ['prob']
    
    processed = []
    for prob_col in prob_columns:
        if prob_col == 'prob':
            class_suffix = 'pred'
            class_name = 'prediction'
        else:
            class_idx = int(prob_col.split('_')[1])
            if class_labels and class_idx in class_labels:
                class_name = class_labels[class_idx]
            else:
                class_name = f'class{class_idx}'
            class_suffix = class_name
        
        output_image_path = os.path.join(output_dir, '{}_{}_{}_{}.png'.format(
            svs_name, suffix, class_suffix, 'heatmap'))
        
        try:
            thumbnail, overlay = create_prediction_overlay(
                svs_file, df_results_sub, prob_column=prob_col, thumbnail_size=thumbnail_size, coord_scale=coord_scale)
            overlay.save(output_image_path, "PNG")
            processed.append(class_name)
        except Exception as e:
            print(f"Error processing {svs_name} for {class_name}: {e}")
    
    return svs_name if processed else None

def main(args):
    thumbnail_size = (1024, 1024)
    wsi_dir = args.wsi_dir
    wsi_list = glob(wsi_dir)

    results_dir = args.results_dir
    output_dir = os.path.join(results_dir, 'visual')
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Found {len(wsi_list)} WSI files")

    results_list = glob(os.path.join(results_dir, '*_inst.csv'))
    print(f"Found {len(results_list)} result CSV files")
    
    if len(results_list) == 0:
        print("Error: No *_inst.csv files found in results directory!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # クラスラベルの解析
    class_labels = None
    if args.class_labels:
        class_labels = {}
        for label_pair in args.class_labels.split(','):
            idx, name = label_pair.split(':')
            class_labels[int(idx)] = name
        print(f"Class labels: {class_labels}")
    
    # Read and process result CSV files
    df_results = []
    for result_path in results_list:
        df = pd.read_csv(result_path)
        print(f"Loaded {result_path}: {len(df)} rows")
        df_results.append(df)
    df_results = pd.concat(df_results, axis=0)
    
    # 数値列のみで平均を取る
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {col: 'mean' for col in numeric_cols}
    df_results = df_results.groupby('filename', as_index=False).agg(agg_dict)
    df_results['svs_name'] = df_results['filename'].map(lambda x: x.split('/')[0])
    
    print(f"Total unique patches: {len(df_results)}")
    print(f"Available columns: {df_results.columns.tolist()}")
    
    num_workers = min(args.num_workers, len(wsi_list))  # Adjust thread count based on CPU cores or user input
        
    # Multithreaded processing of WSIs
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_wsi,
                wsi_list[i],
                df_results,
                output_dir,
                args.model_name,
                thumbnail_size,
                class_labels,
                args.coord_scale
            ): i for i in range(len(wsi_list))
        }
        
        # Track progress with tqdm
        processed_count = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result:
                    processed_count += 1
            except Exception as e:
                print(f"Error processing file: {e}")
        
        print(f"Successfully processed {processed_count} WSIs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process WSIs and generate visual overlays for each class.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model (e.g., smmile).")
    parser.add_argument('--wsi_dir', type=str, required=True, help="Directory pattern for WSI files (e.g., '/path/to/wsi/*.svs').")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing the result CSV files.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for parallel processing.")
    parser.add_argument('--class_labels', type=str, default=None, 
                        help="Class labels in format '0:adenocarcinoma,1:squamous_cell_carcinoma'")
    parser.add_argument('--coord_scale', type=float, default=1.0,
                        help="Coordinate scaling factor to convert CSV coordinates to Level 0 (e.g., 2.0 if coords are at half scale)")
    
    args = parser.parse_args()
    main(args)
