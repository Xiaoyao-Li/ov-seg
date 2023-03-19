import json
import os
from icecream import install
install()

basedir = '/mnt/seagate12t/EPIC-KITCHEN/EPIC-KITCHEN/'
part_list = os.listdir(basedir)
part_list.sort(key=lambda x:int(x[1:]))

info = {p: {} for p in part_list}

for part in part_list:
    ic(part)
    clip_list = os.listdir(os.path.join(basedir, part, 'rgb_frames'))
    clip_list = [f for f in clip_list if os.path.isdir(os.path.join(basedir, part, 'rgb_frames', f))]
    clip_list.sort(key=lambda x:int(x.split('_')[-1]))
    for clip in clip_list:
        frame_dir = os.path.join(basedir, part, 'rgb_frames', clip)
        frame_count = len([f for f in os.listdir(frame_dir)])
        ic(frame_count)
        info[part][clip] = {
            'path': os.path.join(part, 'rgb_frames', clip),
            'count': frame_count
        }
json.dump(info, open(os.path.join(basedir, 'info.json'), 'w'))
