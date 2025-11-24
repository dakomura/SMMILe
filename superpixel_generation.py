import os
import argparse
import glob
import numpy as np
from skimage import segmentation
from tqdm import tqdm
from PIL import Image
import h5py

from concurrent.futures import ThreadPoolExecutor


def load_features_and_coords(fea_path, keyword_feature='feature', target_patch_size=None):
    """
    特徴量と座標を読み込む（h5とnpyの両方に対応）
    
    h5形式（trident出力）の場合、座標はWSI level 0座標で保存されているため、
    ターゲット倍率座標に変換します。
    
    Args:
        fea_path: 特徴量ファイルのパス（.h5 or .npy）
        keyword_feature: npy形式の場合の特徴量キー名
        target_patch_size: h5形式の場合、座標変換後のパッチサイズ（Noneの場合はh5属性から取得）
        
    Returns:
        features: (N, D) ndarray
        coords_nd: (N, 2) ndarray - ターゲット倍率での座標
        inst_label: list or None
        actual_patch_size: 座標系でのパッチサイズ（SMMILeのsize引数に使用）
    """
    file_ext = os.path.splitext(fea_path)[1].lower()
    
    if file_ext == '.h5':
        # trident形式のh5ファイルを読む
        with h5py.File(fea_path, 'r') as f:
            features = f['features'][:]  # (N, D) - e.g., (N, 768) for CONCHv1.5
            coords_level0 = f['coords'][:]  # (N, 2) - level 0座標
            
            # 属性から倍率情報を取得
            if 'coords' in f and hasattr(f['coords'], 'attrs'):
                attrs = dict(f['coords'].attrs)
                patch_size = attrs.get('patch_size', 512)  # ターゲット倍率でのパッチサイズ
                patch_size_level0 = attrs.get('patch_size_level0', patch_size)
                level0_mag = attrs.get('level0_magnification', 40)
                target_mag = attrs.get('target_magnification', 20)
            else:
                # 属性がない場合はデフォルト値を使用
                patch_size = target_patch_size if target_patch_size else 512
                patch_size_level0 = patch_size
                level0_mag = 20
                target_mag = 20
            
            # 座標をターゲット倍率座標に変換
            # coords_target = coords_level0 * target_mag / level0_mag
            scale_factor = target_mag / level0_mag
            coords_nd = (coords_level0 * scale_factor).astype(np.int64)
            
            # 座標系でのパッチサイズ（変換後）
            actual_patch_size = patch_size
            
        inst_label = None  # h5形式にはinst_labelがない
        
    elif file_ext == '.npy':
        # 既存のnpy形式を読む
        record = np.load(fea_path, allow_pickle=True)
        features = record[()][keyword_feature]
        coords = record[()]['index']
        
        # 座標の形式を統一（文字列 or ndarray）
        if isinstance(coords[0], np.ndarray):
            coords_nd = np.array(coords)
        else:
            # "x_y"形式の文字列をパース
            coords_nd = np.array([
                [int(c.split('_')[0]), int(c.split('_')[1])] 
                for c in coords
            ])
        
        # inst_labelの取得（存在しない場合はNone）
        inst_label = record[()].get('inst_label', None)
        
        # npy形式の場合、パッチサイズはファイル名から推測するか引数で指定
        actual_patch_size = target_patch_size if target_patch_size else 512
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .h5 or .npy")
    
    return features, coords_nd, inst_label, actual_patch_size


def get_nic_with_coord(features, coords, size, inst_label):
    w = coords[:,0]
    h = coords[:,1]
    w_min = w.min()
    w_max = w.max()
    h_min = h.min()
    h_max = h.max()
    image_shape = [(w_max-w_min)//size+1,(h_max-h_min)//size+1]
    mask = np.ones((image_shape[0], image_shape[1]))
    labels = -np.ones((image_shape[0], image_shape[1]))
    features_nic = np.ones((features.shape[-1], image_shape[0], image_shape[1])) * np.nan
    coords_nic = -np.ones((image_shape[0], image_shape[1], 2))
    # Store each patch feature in the right position
    if inst_label != [] and inst_label is not None:
        for patch_feature, x, y, label in zip(features, w, h, inst_label):
            coord = [x,y]
            x_nic, y_nic = (x-w_min)//size, (y-h_min)//size
            features_nic[:, x_nic, y_nic] = patch_feature
            coords_nic[x_nic, y_nic] = coord
            labels[x_nic, y_nic]=label
    else:
        for patch_feature, x, y in zip(features, w, h):
            coord = [x,y]
            x_nic, y_nic = (x-w_min)//size, (y-h_min)//size
            features_nic[:, x_nic, y_nic] = patch_feature
            coords_nic[x_nic, y_nic] = coord
        labels=[]

    # Populate NaNs
    mask[np.isnan(features_nic)[0]] = 0
    features_nic[np.isnan(features_nic)] = 0
    return features_nic, mask, labels

def get_sp_label(inst_label_nic, m_slic):
    inst_label_nic = inst_label_nic.T
    
    sp_inst_label_all = []
    
    pred_label_all = []
    
    for i in np.unique(m_slic):
        if i == 0:
            continue
        sp_inst_label = inst_label_nic[m_slic==i]
        sp_inst_label = sp_inst_label.astype(np.int32)
        sp_inst_label[sp_inst_label<0] = 0
        counts = np.bincount(sp_inst_label)
        label = np.argmax(counts)
        
        pred_label_all += [label]*sp_inst_label.shape[0]
        
        sp_inst_label_all += list(sp_inst_label)

    return sp_inst_label_all, pred_label_all

def generate_adjacency_matrix(superpixel_matrix):
    # 获取超像素矩阵的形状
    rows, cols = superpixel_matrix.shape
    num_superpixels = np.max(superpixel_matrix)+1

    # 创建邻接矩阵，并初始化为0
    adjacency_matrix = np.zeros((num_superpixels, num_superpixels), dtype=int)

    # 遍历超像素矩阵的每个像素
    for i in range(rows):
        for j in range(cols):
            current_superpixel = superpixel_matrix[i, j]
            
            if current_superpixel == 0:
                continue

            # 检查当前像素的上方和左方是否与当前超像素相邻
            if i > 0 and superpixel_matrix[i-1, j] != current_superpixel:
                adjacency_matrix[current_superpixel, superpixel_matrix[i-1, j]] = 1
                adjacency_matrix[superpixel_matrix[i-1, j], current_superpixel] = 1

            if j > 0 and superpixel_matrix[i, j-1] != current_superpixel:
                adjacency_matrix[current_superpixel, superpixel_matrix[i, j-1]] = 1
                adjacency_matrix[superpixel_matrix[i, j-1], current_superpixel] = 1

    return adjacency_matrix

def process_file(fea_path, args):
    base_name = os.path.basename(fea_path)
    file_ext = os.path.splitext(fea_path)[1].lower()
    
    # 特徴量と座標を読み込む（h5/npy両対応、座標変換込み）
    features, coords_nd, inst_label, actual_patch_size = load_features_and_coords(
        fea_path, 
        keyword_feature=args.keyword_feature,
        target_patch_size=args.size
    )
    
    # inst_labelがNoneの場合はダミーラベルを作成
    if inst_label is None:
        inst_label = [0 for _ in range(coords_nd.shape[0])]
    
    # sizeはh5から取得したactual_patch_sizeを使用（h5の場合）
    # npy形式または明示的に指定された場合はargs.sizeを使用
    size_to_use = actual_patch_size if file_ext == '.h5' else args.size
    
    if args.debug:
        print(f"Processing: {fea_path}")
        print(f"  Features shape: {features.shape}")
        print(f"  Coords shape: {coords_nd.shape}")
        print(f"  Coords range: x={coords_nd[:,0].min()}-{coords_nd[:,0].max()}, y={coords_nd[:,1].min()}-{coords_nd[:,1].max()}")
        print(f"  Size used: {size_to_use}")
        print(f"  Inst label sample: {inst_label[:5] if inst_label else None}")

    features = (features - features.min()) / (features.max() - features.min()) * 255
    features_nic, mask, _ = get_nic_with_coord(features, coords_nd, size_to_use, inst_label)

    data = np.transpose(features_nic, (1, 2, 0))
    n_segments = max(10, int(features.shape[0] / args.n_segments_persp))
    m_slic = segmentation.slic(data, n_segments=n_segments, mask=mask, compactness=args.compactness, start_label=1).T
    m_adj = generate_adjacency_matrix(m_slic)

    # 出力ファイル名を生成（h5の場合は.h5を.npyに変換）
    if file_ext == '.h5':
        out_name = base_name.replace('.h5', '.npy')
    else:
        out_name = base_name
    
    sp_path = os.path.join(args.sp_dir, out_name)
    sp_contents = {'m_slic': m_slic, 'm_adj': m_adj}
    
    if np.max(m_slic) != (m_adj.shape[0] - 1):
        print(f"Warning: Inconsistent superpixel count in {fea_path}")
        print(f"  m_slic max: {np.max(m_slic)}, m_adj shape: {m_adj.shape}")

    np.save(sp_path, sp_contents)

def main():
    parser = argparse.ArgumentParser(description="Process WSIs for Superpixel Segmentation")
    parser.add_argument('--size', type=int, default=512, 
                        help='Patch size for NIC conversion (default: 512). For h5 files, this is auto-detected from file attributes.')
    parser.add_argument('--file_suffix', type=str, default='*_0_512.h5', 
                        help='Suffix for h5/npy files (e.g., *_0_512.h5 or *_1_256.npy)')
    parser.add_argument('--n_segments_persp', type=int, default=16, 
                        help='Number of patches per super-patch')
    parser.add_argument('--keyword_feature', type=str, default='feature', 
                        help='Feature keyword in npy file (ignored for h5 files)')
    parser.add_argument('--compactness', type=int, default=50, 
                        help='Compactness for SLIC (default: 50)')
    parser.add_argument('--fea_dir', type=str, required=True,
                        help='Directory for feature embeddings of WSIs')
    parser.add_argument('--sp_dir', type=str, default=None,
                        help='Directory for saving superpixel segmentation results. If not specified, auto-generated.')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker threads')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()
    
    # sp_dirが指定されていない場合は自動生成
    if args.sp_dir is None:
        args.sp_dir = os.path.join(
            os.path.dirname(args.fea_dir.rstrip('/')),
            f'sp_n{args.n_segments_persp}_c{args.compactness}_{args.size}'
        )

    print("Superpixel Segmentation Results will be saved to: %s" % args.sp_dir)

    file_list = glob.glob(os.path.join(args.fea_dir, args.file_suffix))
    print(f"Found {len(file_list)} files matching pattern: {args.file_suffix}")

    if len(file_list) == 0:
        print("No files found. Check the directory path and file suffix.")
        return

    if not os.path.exists(args.sp_dir):
        os.makedirs(args.sp_dir)

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(lambda fea_path: process_file(fea_path, args), file_list), total=len(file_list)))

if __name__ == "__main__":
    main()
