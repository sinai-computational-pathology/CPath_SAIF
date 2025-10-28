import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py

def login_to_dsa_api(base_url, username, password, otp=None):
    """
    Log in to the Digital Slide Archive API using HTTP Basic Auth and return the authenticated session and Girder-Token.
    """
    login_url = f"{base_url}/user/authentication"
    session = requests.Session()
    headers = {}
    if otp:
        headers['Girder-OTP'] = otp
    response = session.get(login_url, auth=(username, password), headers=headers)
    response.raise_for_status()
    login_json = response.json()
    girder_token = login_json.get('authToken', {}).get('token')
    if not girder_token:
        raise Exception("No Girder-Token found in login response!")
    session.headers.update({'Girder-Token': girder_token})
    return session, girder_token

def get_embedding_path(minerva_path, magnification, patch_size):
    batch_assignment_df = pd.read_csv("/hpc/users/chens61/comppath_SAIF/data/make_features_trident/slide_batch_assignment.csv")
    batch_rows = batch_assignment_df.loc[batch_assignment_df['minerva_path'] == minerva_path, 'batch_number']
    if batch_rows.empty:
        return None, None
    batch = batch_rows.values[0]
    slide_name = os.path.basename(minerva_path).replace('.tiff', '')
    base_folder = "/hpc/users/chens61/comppath_SAIF/data/make_features_trident/trident_features"
    folder = f"{base_folder}/batch_{batch:04d}/trident_processed/{magnification}_{patch_size}px_0px_overlap/features_sp22m"
    return os.path.join(folder, f"{slide_name}.h5"), batch

def get_leaf_folders(session, base_url, parent_id, parent_type='collection'):
    """
    Recursively find all leaf folders (folders with no subfolders) under the given parent.
    """
    folder_url = f"{base_url}/folder"
    params = {
        "parentType": parent_type,
        "parentId": parent_id,
        "limit": 50,
        "sort": "lowerName",
        "sortdir": 1
    }
    r = session.get(folder_url, params=params)
    r.raise_for_status()
    folders = r.json()
    leaf_folders = []
    for folder in folders:
        # Check for subfolders
        subfolder_params = {
            "parentType": "folder",
            "parentId": folder['_id'],
            "limit": 1
        }
        sub_r = session.get(folder_url, params=subfolder_params)
        sub_r.raise_for_status()
        subfolders = sub_r.json()
        if subfolders:
            # Recurse into subfolders
            leaf_folders.extend(get_leaf_folders(session, base_url, folder['_id'], parent_type='folder'))
        else:
            leaf_folders.append(folder)
    return leaf_folders

if __name__ == "__main__":
    BASE_URL = "https://pathologywsi.hpc.mssm.edu/api/v1"
    import getpass
    USERNAME = os.environ.get("DSA_USERNAME")
    PASSWORD = os.environ.get("DSA_PASSWORD")
    if not USERNAME:
        USERNAME = input("Enter your username: ").strip()
    if not PASSWORD:
        PASSWORD = getpass.getpass("Enter your password: ")
    try:
        session, girder_token = login_to_dsa_api(BASE_URL, USERNAME, PASSWORD)
    except requests.HTTPError as e:
        print("Login failed! Please check your username and password.")
        exit(1)
    except Exception as e:
        print(f"Login failed: {e}")
        exit(1)

    # === Preset variables for easy modification ===
    collection_id = '68b86487bf0005b1fefe0095'
    tile_size = 224  # default tile size for prediction heatmap
    current_group_setting = 'gma_sex_081425'  # group name for prediction heatmap
    mag = 20  # magnification used for tile extraction
    # mpp_base = 0.25 for 40x, so scale by (40 / mag)
    tile_width = tile_size * (40 / mag)
    tile_height = tile_size * (40 / mag)

    print(f"Finding all leaf folders in collection {collection_id}...")
    leaf_folders = get_leaf_folders(session, BASE_URL, collection_id, parent_type='collection')
    print(f"Found {len(leaf_folders)} leaf folders.")

    all_items = []
    for folder in leaf_folders:
        folder_id = folder['_id']
        item_url = f"{BASE_URL}/item"
        item_params = {"folderId": folder_id}
        ir = session.get(item_url, params=item_params)
        ir.raise_for_status()
        items = ir.json()
        for item in items:
            all_items.append(item)

    print(f"Found {len(all_items)} items (slides) in all leaf folders.")

    def upload_attention_map_from_high_conf_df(session, BASE_URL, high_conf_csv_path, all_items):
        print(f"Step 6: Uploading attention maps from {high_conf_csv_path}...")
        df = pd.read_csv(high_conf_csv_path)
        # Build mapping from slide_name (split + .tiff) to item
        item_name_to_item = {item['name']: item for item in all_items}
        # print("all_items names:", list(item_name_to_item.keys()))
        for idx, row in df.iterrows():
            slide = row['slide']
            slide_name = row['slide_name'].split('_')[0] + '.tiff'
            print(f"Trying slide_name: {slide_name}")
            seed = row['seed']
            minerva_path = row['minerva_path']
            embedding_path, batch = get_embedding_path(minerva_path, f"{mag}x", tile_size)
            with h5py.File(embedding_path, 'r') as f:
                coords = f['coords'][:]
            attn_map_path = f"/hpc/users/chens61/comppath_SAIF/slide_experiments/sex/Colon/mccv_081425/case/all/seed_{seed}/attention_scores_dict.pkl"
            if not os.path.exists(attn_map_path):
                print(f"  Attention map not found for seed {seed}: {attn_map_path}")
                continue
            with open(attn_map_path, 'rb') as f:
                attn_dict = pickle.load(f)
            if slide not in attn_dict:
                print(f"  Slide {slide} not found in attention map {attn_map_path}")
                continue
            attn_scores = attn_dict[slide]  # shape: [number of tiles, 2] or [number of tiles]
            for class_idx, class_name in enumerate(["Male", "Female"]):
                elements = []
                scores = attn_scores[:, class_idx]
                vmin = float(scores.min())
                vmax = float(scores.max())
                cmap = plt.get_cmap('viridis')
                def prob_to_color(val):
                    norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                    rgba = cmap(norm_val)
                    r, g, b = [int(255 * x) for x in rgba[:3]]
                    return f"rgb({r},{g},{b})"
                for i in range(coords.shape[0]):
                    x, y = float(coords[i, 0]), float(coords[i, 1])
                    center_x = x + float(tile_width) / 2
                    center_y = y + float(tile_height) / 2
                    score = float(scores[i])
                    fill_color = prob_to_color(score)
                    rect = {
                        "type": "rectangle",
                        "center": [center_x, center_y, 0],
                        "width": float(tile_width),
                        "height": float(tile_height),
                        "rotation": 0,
                        "normal": [0, 0, 1],
                        "fillColor": fill_color,
                        "lineColor": "rgba(0,0,0,0)",
                        "lineWidth": 0,
                        "label": {"value": f"attn_score={score:.3f}"},
                        "group": "Sex_Prediction"
                    }
                    elements.append(rect)
                annotation_json = {
                    "name": f"SAIF Attention Map ({class_name})",
                    "description": f"Attention map for slide {slide_name}, seed={seed}, class={class_name}",
                    "elements": elements
                }
                item = item_name_to_item.get(slide_name)
                if not item:
                    print(f"  Item not found for slide_name {slide_name}")
                    continue
                annotation_url = f"{BASE_URL}/annotation"
                params = {"itemId": item['_id']}
                headers = {"Content-Type": "application/json"}
                resp = session.post(annotation_url, params=params, json=annotation_json, headers=headers)
                if resp.status_code == 200:
                    print(f"  Attention map uploaded for {slide_name} ({class_name})!")
                else:
                    print(f"  Failed to upload attention map for {slide_name} ({class_name}): {resp.status_code}")
                    print(resp.text)
            # break  # Only upload for the first row in the DataFrame

    high_conf_csv_path = '/hpc/users/chens61/comppath_SAIF/slide_experiments/sex/Colon/mccv_081425/.DSA_Uploader/090325/high_conf_df.csv'
    upload_attention_map_from_high_conf_df(session, BASE_URL, high_conf_csv_path, all_items)