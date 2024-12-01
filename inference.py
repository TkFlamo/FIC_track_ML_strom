import argparse
import copy
import os
import shutil
from time import time
from typing import List, Dict, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from torch import nn
from scipy.optimize import linear_sum_assignment

from osnet import osnet_x1_0
from utils import load_checkpoint, copy_state_dict


SHOW_PREDICTIONS = False


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', '-v', type=str, required=True, help='Path to directory with test videos.')
    parser.add_argument('--mount', '-m', type=str, required=True, help='Path to directory with trained models and other required data.')
    parser.add_argument('--save_dir', '-s', type=str, required=True, help='Path to directory where predictions will be saved.')
    return parser.parse_args()


def get_avg_fps(processing_times: List[float]) -> float:
    return 1 / np.mean(processing_times)


def create_reid_model(checkpoint: str='assets/osnet.pth'):
    '''
    Возвращает модель для REID.
    '''
    reid_model = osnet_x1_0(num_classes=751, chkpt_path=checkpoint)
    reid_model.cuda()
    reid_model = nn.DataParallel(reid_model)
    return reid_model


def clean_dist_matrix(matrix: np.ndarray, value: float = 100.0):
    non_100_rows = (matrix != value).any(dim=1)
    matrix = matrix[non_100_rows]

    non_100_cols = (matrix != value).any(dim=0)
    matrix = matrix[:, non_100_cols]

    return matrix


class MCMOT:
    def __init__(
        self,
        person_detector: YOLO,
        feature_extractor: nn.Module,
        num_cams: int,
        similarity_threshold: float,
        log_dir: str,
        tracker_path: str,
    ):
        self._detectors = [copy.deepcopy(person_detector) for _ in range(num_cams)]

        self._feature_extractor = feature_extractor
        self._feature_extractor.eval()
        self._fe_transforms = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._similarity_threshold = similarity_threshold
        self._log_dir = log_dir
        self._tracker_path = tracker_path
        self._num_cams = num_cams

        self._global_id = 1000  
        self._local_id_2_global_id = {}  

    def track(self, timestamp: int, frames: List[np.ndarray], log_filenames: List[str]) -> List[np.ndarray]:
        frames_embeddings, frames_bboxes = self._get_persons_embeddings(frames)
        if frames_embeddings != [{}] * self._num_cams:
            matches = self._get_matches(frames_embeddings)
            frames_bboxes = self._get_global_matches(matches, frames_bboxes)
            det_frames = self._vizualize(frames, frames_bboxes)
            self._write_logs(timestamp, frames_bboxes, log_filenames)
            return det_frames
        return [(f'cam # {cam_id}', frame) for cam_id, frame in enumerate(frames)]

    def _get_persons_embeddings(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[List[Dict[int, np.ndarray]], List[Dict[int, Tuple[int, int, int, int]]]]:
        '''
        Трекинг людей на кадре и получение эмбеддингов.

        frames: List[np.ndarray]
            Кадры с разных камер.
        '''
        frames_embeddings = []
        frames_bboxes = []
        for cam_id, frame in enumerate(frames):
            results = self._detectors[cam_id].track(frame, persist=True, classes=[0], verbose=True, tracker=self._tracker_path)
            boxes = results[0].boxes
            frames_embeddings.append(self._get_features(frame, boxes))
            frames_bboxes.append(self._get_bboxes(boxes))
        return frames_embeddings, frames_bboxes

    def _get_features(
        self,
        frame: np.ndarray,
        boxes,
    ) -> List[np.ndarray]:
        persons_imgs = []
        persons_ids = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().int().numpy().reshape(-1)
            person_img = cv2.resize(frame[y1:y2, x1: x2, :], (256, 128))
            persons_imgs.append(person_img)
            try:
                track_id = box.id.int().item()
            except AttributeError:
                track_id = -1
            persons_ids.append(track_id)
        
        if not persons_imgs:
            return {}
        
        persons_imgs = np.stack(persons_imgs, axis=0)
        persons_imgs = torch.tensor(persons_imgs.transpose(0, 3, 1, 2)) / 255.
        persons_imgs = self._fe_transforms(persons_imgs)

        with torch.no_grad():
            embeddings = self._feature_extractor(persons_imgs)

        return {
            track_id: embedding.cpu().numpy()
            for track_id, embedding in zip(persons_ids, embeddings)
        }      

    def _get_bboxes(
        self,
        boxes,
    ) -> Dict[int, Tuple[int, int, int, int]]:
        bboxes = {}
        for box in boxes:
            try:
                track_id = box.id.int().item()
            except AttributeError:
                track_id = -1
            bboxes[track_id] = box.xyxy.cpu().int().numpy().reshape(-1).tolist() 
        return bboxes

    def _get_matches(self, frames_embeddings: List[Dict[int, np.ndarray]]) -> List[Tuple[int, int]]:
        all_embeddings = []
        ids = []
        frames_tracks_cnt = []
        for frame_id, frame_embedding in enumerate(frames_embeddings):
            all_embeddings.extend(frame_embedding.values())
            ids.extend([f'{frame_id}-{track_id}' for track_id in frame_embedding.keys()])
            frames_tracks_cnt.append(len(frame_embedding))
        
        n = len(all_embeddings)
        all_embeddings = [torch.tensor(embedding) for embedding in all_embeddings]
        x = torch.cat(all_embeddings)
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        for i, frame_tracks_cnt in enumerate(frames_tracks_cnt):
            if i == 0:
                dist_m[:frame_tracks_cnt, :frame_tracks_cnt] = 100
            else:
                dist_m[
                    frames_tracks_cnt[i-1]: (frames_tracks_cnt[i-1] + frame_tracks_cnt),
                    frames_tracks_cnt[i-1]: (frames_tracks_cnt[i-1] + frame_tracks_cnt)
                ] = 100
        
        dist_m[dist_m > self._similarity_threshold] = 100

        row_ind, col_ind = linear_sum_assignment(dist_m)
        pairs = list(zip(row_ind, col_ind))
        values = dist_m[row_ind, col_ind].tolist()

        matches = []
        for pair, value in zip(pairs, values):
            if pair[0] <= pair[1] and value <= self._similarity_threshold:
                matches.append((ids[pair[0]], ids[pair[1]]))
        
        return matches

    def _get_global_matches(self, matches: List[Tuple[str, str]], frames_bboxes: List[Dict[int, Tuple[int, int, int, int]]]):
        for match in matches:
            if match[0] not in self._local_id_2_global_id.keys() and match[1] not in self._local_id_2_global_id.keys():
                self._local_id_2_global_id[match[0]] = self._global_id
                self._local_id_2_global_id[match[1]] = self._global_id
                self._global_id += 1
            elif match[0] in self._local_id_2_global_id.keys() and match[1] not in self._local_id_2_global_id.keys():
                self._local_id_2_global_id[match[1]] = self._local_id_2_global_id[match[0]]
            elif match[1] in self._local_id_2_global_id.keys() and match[0] not in self._local_id_2_global_id.keys():
                self._local_id_2_global_id[match[0]] = self._local_id_2_global_id[match[1]]
        
        for camera_id, frame_bboxes in enumerate(frames_bboxes):
            for track_id in list(frame_bboxes.keys()):
                id_code = f'{camera_id}-{track_id}'
                if id_code in self._local_id_2_global_id:
                    new_track_id = self._local_id_2_global_id[id_code]
                    frame_bboxes[new_track_id] = frame_bboxes.pop(track_id)
        return frames_bboxes

    def _vizualize(self, frames: List[np.ndarray], frames_bboxes: List[Dict[int, Tuple[int, int, int, int]]]):
        det_frames = []
        for cam_i, (frame, frame_bboxes) in enumerate(zip(frames, frames_bboxes)):
            det_frame = frame.copy()  
            for object_id, object_bbox in frame_bboxes.items():
                det_frame = cv2.rectangle(det_frame, object_bbox[:2], object_bbox[2:], (0, 0, 0), 4)
                det_frame = cv2.rectangle(det_frame, object_bbox[:2], object_bbox[2:], (0, 0, 250), 2)
                det_frame = cv2.putText(det_frame, f"ID: {object_id}", object_bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 1.02, (0, 0, 0), 2)
                det_frame = cv2.putText(det_frame, f"ID: {object_id}", object_bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
            det_frames.append((f'cam # {cam_i}', det_frame))
        return det_frames

    def _write_logs(self, timestamp: int, frames_bboxes: List[Dict[int, Tuple[int, int, int, int]]], log_filenames: List[str]):
        for frame_bboxes, log_filename in zip(frames_bboxes, log_filenames):
            with open(os.path.join(self._log_dir, log_filename), 'a') as log:
                for track_id, bbox in frame_bboxes.items():
                    x1, y1, x2, y2 = bbox
                    log.write(f"{timestamp}, {track_id}, {x1}, {y1}, {x2-x1}, {y2-y1}, 1, 1, 1, 1\n")


if __name__ == '__main__':
    args = parse()

    processing_times = []

    reid_model_checkpoint = os.path.join(
        args.mount,
        'osnet.pth',
    )

    videos_list = Path(args.videos_dir).rglob('*.mp4')
    videos_list = sorted(videos_list)

    mcmot = MCMOT(
        person_detector=YOLO(os.path.join(args.mount, "yolo11m.pt")),
        feature_extractor=create_reid_model(reid_model_checkpoint),
        similarity_threshold=0.5,
        num_cams=len(videos_list),
        log_dir=args.save_dir,
        tracker_path=os.path.join(args.mount, 'bytetrack.yaml')
    )

    caps = [cv2.VideoCapture(video_path) for video_path in videos_list]
    videos_basenames = [os.path.basename(video_path) for video_path in videos_list]
    video_logs = [os.path.splitext(video_name)[0] + '.txt' for video_name in videos_basenames]

    assert args.save_dir != './' and args.save_dir != '.'
    if os.path.exists(args.save_dir) and os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    ret = False
    frames = []
    timestamp = 1
    for cap in caps:
        cap_ret, frame = cap.read()
        frames.append(frame)
        ret = ret or cap_ret

    while ret:
        start_time = time()
        det_frames = mcmot.track(timestamp, frames, video_logs)
        end_time = time()

        process_time = end_time - start_time
        processing_times.append(process_time)

        if SHOW_PREDICTIONS:
            for title, det_frame in det_frames:
                cv2.imshow(title, det_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        frames = []
        for cap in caps:
            cap_ret, frame = cap.read()
            frames.append(frame)
            ret = ret and cap_ret
        timestamp += 1

    avg_fps = get_avg_fps(processing_times)
    print(f'Average FPS: {avg_fps}')
