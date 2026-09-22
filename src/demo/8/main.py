import pandas as pd
import numpy as np
import os
from pathlib import Path


all_files = {
    "P2": ["classic_white_seed5_2.csv", "view_white_seed4_2.csv", "classic_blur_seed3_2.csv", "view_blur_seed2_2.csv"],
    "P3": ["classic_white_ seed5_3.csv", "view_white_seed4_3.csv", "classic_blur_seed3_3.csv", "view_blur_seed2_3.csv"],
    "P4": ["system_log_20260917211910.csv", "system_log_20260917211625.csv"],
    "P5": ["system_log_20260917212841.csv", "system_log_20260917212627.csv"],
    "P6": ["system_log_20260922155712.csv", "system_log_20260922160033.csv"],
    "P7": ["system_log_20260922160916.csv", "system_log_20260922161123.csv"],
}

def get_condition_from_msg(msg):
    anim = 'classic' if 'type=classic' in str(msg) else 'view'
    backdrop = 'white' if 'backdrop=white' in str(msg) else 'blur'
    return anim, backdrop

results = []

for part_id, files in all_files.items():
    for file in files:
        file_path = Path.joinpath(Path(__file__).parent, "_data", part_id, file)

        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Get condition
            start_row = df[df['tag'] == 'start']
            msg = start_row['message'].iloc[0]
            anim, backdrop = get_condition_from_msg(msg)
                
            cond = f"{anim}_{backdrop}"
            
            # Times
            comp_row = df[df['tag'] == 'complete']
            if not start_row.empty and not comp_row.empty:
                task_time_sec = (comp_row['timestamp'].max() - start_row['timestamp'].min()) / 1000.0
            else:
                task_time_sec = np.nan
                
            # Modals and deletes
            modal_opens = len(df[df['tag'] == 'open_modal'])
            deletes = len(df[df['tag'] == 'delete'])
            
            # Gallery Analysis (Time & Mouse Dist)
            gallery_times = []
            gallery_mouse_dists = []
            last_close_time = None
            
            for i, row in df.iterrows():
                tag = row['tag']
                t = row['timestamp']
                if tag == 'start' or tag in ['close_modal', 'delete']:
                    last_close_time = t
                elif tag == 'open_modal':
                    if last_close_time is not None:
                        duration = t - last_close_time
                        gallery_times.append(duration)
                        
                        mouse_moves = df[(df['timestamp'] >= last_close_time) & (df['timestamp'] <= t) & (df['tag'] == 'mouse_move')]
                        dist = 0
                        if len(mouse_moves) > 1:
                            dx = mouse_moves['x'].diff()
                            dy = mouse_moves['y'].diff()
                            dist = np.sqrt(dx**2 + dy**2).sum()
                        gallery_mouse_dists.append(dist)
                    last_close_time = None
                    
            # Error Analysis (Coordinate based)
            start_t = start_row['timestamp'].min() if not start_row.empty else df['timestamp'].min()
            end_t = comp_row['timestamp'].max() if not comp_row.empty else df['timestamp'].max()
            df_task = df[(df['timestamp'] >= start_t) & (df['timestamp'] <= end_t)]
            
            opens = df_task[df_task['tag'] == 'open_modal'].copy()
            def get_col(x):
                if x < 320: return 0
                elif x < 640: return 1
                elif x < 960: return 2
                elif x < 1280: return 3
                elif x < 1600: return 4
                else: return 5
            def get_row(y):
                if y < 400: return 0
                elif y < 700: return 1
                else: return 2
                
            opens['col'] = opens['x'].apply(get_col)
            opens['row'] = opens['y'].apply(get_row)
            opens['card_id'] = opens['row'] * 6 + opens['col']
            
            errors = 0
            error_details = []
            counts = opens['card_id'].value_counts()
            dup_cards = counts[counts > 1].index.tolist()
            
            for card in dup_cards:
                times = opens[opens['card_id'] == card]['timestamp'].tolist()
                for t in times[1:]:
                    prev_events = df_task[(df_task['timestamp'] < t) & (~df_task['tag'].isin(['mouse_move']))].tail(2)
                    if not any(r['tag'] == 'delete' for _, r in prev_events.iterrows()):
                        errors += 1
                        error_details.append(card)
                        
            results.append({
                'Participant': part_id,
                'Anim': anim,
                'Backdrop': backdrop,
                'Task_Time_s': task_time_sec,
                'Modal_Opens': modal_opens,
                'Deletes': deletes,
                'Avg_Gallery_Dist_px': np.mean(gallery_mouse_dists) if gallery_mouse_dists else 0,
                'Avg_Gallery_Time_ms': np.mean(gallery_times) if gallery_times else 0,
                'Errors': errors
            })
        except Exception as e:
            print(f"Failed on {file_path}: {e}")

df_res = pd.DataFrame(results)

print("--- 全被験者・全セッションの一覧データ ---")
print(df_res.sort_values(by=['Participant', 'Anim', 'Backdrop']).to_markdown(index=False))

print("\n--- アニメーション別の総合平均 (N=16セッション) ---")
agg_anim = df_res.groupby('Anim')[['Task_Time_s', 'Modal_Opens', 'Avg_Gallery_Dist_px', 'Avg_Gallery_Time_ms', 'Errors']].mean().reset_index()
print(agg_anim.to_markdown(index=False))

print("\n--- アニメーション × バックドロップの総合平均 ---")
agg_cond = df_res.groupby(['Anim', 'Backdrop'])[['Task_Time_s', 'Modal_Opens', 'Avg_Gallery_Dist_px', 'Avg_Gallery_Time_ms', 'Errors']].mean().reset_index()
print(agg_cond.to_markdown(index=False))

print("\n--- エラー発生状況のサマリ ---")
err_summary = df_res.groupby('Anim')['Errors'].sum().reset_index()
print(err_summary.to_markdown(index=False))