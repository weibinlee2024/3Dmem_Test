import os
import json
import pickle
from collections import defaultdict
import logging
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union

import habitat_sim

from src.scene_goatbench import Scene
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from PIL import Image

        

class Logger:
    def __init__(
        self,
        output_dir,
        start_ratio,
        end_ratio,
        split,
        voxel_size,  # used for calculating the moving distance
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size

        if os.path.exists(
            os.path.join(
                output_dir, f"success_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.success_by_snapshot = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"success_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.success_by_snapshot = {}  # subtask_id -> success
        if os.path.exists(
            os.path.join(
                output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.success_by_distance = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.success_by_distance = {}  # subtask id -> success
        if os.path.exists(
            os.path.join(
                output_dir, f"spl_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.spl_by_snapshot = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"spl_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.spl_by_snapshot = {}  # subtask id -> spl
        if os.path.exists(
            os.path.join(
                output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.spl_by_distance = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.spl_by_distance = {}  # subtask id -> spl
        if os.path.exists(
            os.path.join(
                output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.success_by_task = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            # success_by_task = {}  # task type -> success
            self.success_by_task = defaultdict(list)
        if os.path.exists(
            os.path.join(
                output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.spl_by_task = pickle.load(
                open(
                    os.path.join(
                        output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
                    ),
                    "rb",
                )
            )
        else:
            # spl_by_task = {}  # task type -> spl
            self.spl_by_task = defaultdict(list)
        if os.path.exists(
            os.path.join(
                output_dir,
                f"n_filtered_snapshots_{start_ratio}_{end_ratio}_{split}.json",
            )
        ):
            with open(
                os.path.join(
                    output_dir,
                    f"n_filtered_snapshots_{start_ratio}_{end_ratio}_{split}.json",
                ),
                "r",
            ) as f:
                self.n_filtered_snapshots_list = json.load(f)
        else:
            self.n_filtered_snapshots_list = {}
        if os.path.exists(
            os.path.join(
                output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}_{split}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir,
                    f"n_total_snapshots_{start_ratio}_{end_ratio}_{split}.json",
                ),
                "r",
            ) as f:
                self.n_total_snapshots_list = json.load(f)
        else:
            self.n_total_snapshots_list = {}
        if os.path.exists(
            os.path.join(
                output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"
                ),
                "r",
            ) as f:
                self.n_total_frames_list = json.load(f)
        else:
            self.n_total_frames_list = {}

        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.split = split

        # some sanity check
        assert (
            len(self.success_by_snapshot)
            == len(self.spl_by_snapshot)
            == len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_snapshot)} != {len(self.spl_by_snapshot)} != {len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
            == len(self.success_by_snapshot)
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])} != {len(self.success_by_snapshot)}"

        # logging for episode
        self.episode_dir = None

        # logging for subtask
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def save_results(self):
        # some sanity check
        assert (
            len(self.success_by_snapshot)
            == len(self.spl_by_snapshot)
            == len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_snapshot)} != {len(self.spl_by_snapshot)} != {len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
            == len(self.success_by_snapshot)
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])} != {len(self.success_by_snapshot)}"

        with open(
            os.path.join(
                self.output_dir,
                f"success_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_by_snapshot, f)
        with open(
            os.path.join(
                self.output_dir,
                f"success_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_by_distance, f)
        with open(
            os.path.join(
                self.output_dir,
                f"spl_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.spl_by_snapshot, f)
        with open(
            os.path.join(
                self.output_dir,
                f"spl_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.spl_by_distance, f)
        with open(
            os.path.join(
                self.output_dir,
                f"success_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_by_task, f)
        with open(
            os.path.join(
                self.output_dir,
                f"spl_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.spl_by_task, f)
        with open(
            os.path.join(
                self.output_dir,
                f"n_filtered_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_filtered_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_frames_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_frames_list, f, indent=4)

    def aggregate_results(self):
        # aggregate the results into a single file
        filenames_to_merge = [
            "success_by_snapshot",
            "spl_by_snapshot",
            "success_by_distance",
            "spl_by_distance",
        ]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    all_results.update(pickle.load(f))
            logging.info(
                f"Total {filename} results: {100 * np.mean(list(all_results.values())):.2f}, len: {len(all_results)}"
            )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)
        filenames_to_merge = ["success_by_task", "spl_by_task"]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    separate_stat = pickle.load(f)
                    for task_name, task_res in separate_stat.items():
                        if task_name not in all_results:
                            all_results[task_name] = []
                        all_results[task_name] += task_res
            for task_name, task_res in all_results.items():
                logging.info(
                    f"Total {filename} results for {task_name}: {100 * np.mean(task_res):.2f}, len: {len(task_res)}"
                )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)

        n_filtered_snapshots_list = {}
        all_n_filtered_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_filtered_snapshots_*.json")
        )
        for n_filtered_snapshots_list_path in all_n_filtered_snapshots_list_paths:
            with open(n_filtered_snapshots_list_path, "r") as f:
                n_filtered_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_filtered_snapshots.json"), "w") as f:
            json.dump(n_filtered_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of filtered snapshots: {np.mean(list(n_filtered_snapshots_list.values()))}"
        )

        n_total_snapshots_list = {}
        all_n_total_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_snapshots_*.json")
        )
        for n_total_snapshots_list_path in all_n_total_snapshots_list_paths:
            with open(n_total_snapshots_list_path, "r") as f:
                n_total_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_snapshots.json"), "w") as f:
            json.dump(n_total_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of total snapshots: {np.mean(list(n_total_snapshots_list.values()))}"
        )

        n_total_frames_list = {}
        all_n_total_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_frames_*.json")
        )
        for n_total_frames_list_path in all_n_total_frames_list_paths:
            with open(n_total_frames_list_path, "r") as f:
                n_total_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_frames.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)
        logging.info(
            f"Average number of total frames: {np.mean(list(n_total_frames_list.values()))}"
        )

    def log_subtask_result(
        self,
        success_by_snapshot: bool,
        success_by_distance: bool,
        subtask_id: str,
        gt_subtask_explore_dist: float,
        goal_type: str,
        n_filtered_snapshots,
        n_total_snapshots,
        n_total_frames,
    ):
        if success_by_snapshot:
            self.success_by_snapshot[subtask_id] = 1.0
        else:
            self.success_by_snapshot[subtask_id] = 0.0
        if success_by_distance:
            self.success_by_distance[subtask_id] = 1.0
        else:
            self.success_by_distance[subtask_id] = 0.0

        # calculate the SPL
        self.spl_by_snapshot[subtask_id] = (
            self.success_by_snapshot[subtask_id]
            * gt_subtask_explore_dist
            / max(gt_subtask_explore_dist, self.subtask_explore_dist)
        )
        self.spl_by_distance[subtask_id] = (
            self.success_by_distance[subtask_id]
            * gt_subtask_explore_dist
            / max(gt_subtask_explore_dist, self.subtask_explore_dist)
        )

        self.success_by_task[goal_type].append(self.success_by_distance[subtask_id])
        self.spl_by_task[goal_type].append(self.spl_by_distance[subtask_id])

        logging.info(
            f"Subtask {subtask_id} finished, {self.subtask_explore_dist} length"
        )
        logging.info(
            f"Subtask spl by snapshot: {self.spl_by_snapshot[subtask_id]}, spl by distance: {self.spl_by_distance[subtask_id]}"
        )

        logging.info(
            f"Success rate by snapshot: {100 * np.mean(np.asarray(list(self.success_by_snapshot.values()))):.2f}"
        )
        logging.info(
            f"Success rate by distance: {100 * np.mean(np.asarray(list(self.success_by_distance.values()))):.2f}"
        )
        logging.info(
            f"SPL by snapshot: {100 * np.mean(np.asarray(list(self.spl_by_snapshot.values()))):.2f}"
        )
        logging.info(
            f"SPL by distance: {100 * np.mean(np.asarray(list(self.spl_by_distance.values()))):.2f}"
        )

        for task_name, success_list in self.success_by_task.items():
            logging.info(
                f"Success rate for {task_name}: {100 * np.mean(np.asarray(success_list)):.2f}"
            )
        for task_name, spl_list in self.spl_by_task.items():
            logging.info(
                f"SPL for {task_name}: {100 * np.mean(np.asarray(spl_list)):.2f}"
            )

        logging.info(
            f"Filtered snapshots/Total snapshots/Total frames: {n_filtered_snapshots}/{n_total_snapshots}/{n_total_frames}"
        )
        # save the number of snapshots
        self.n_filtered_snapshots_list[subtask_id] = n_filtered_snapshots
        self.n_total_snapshots_list[subtask_id] = n_total_snapshots
        self.n_total_frames_list[subtask_id] = n_total_frames

        # clear the subtask logging
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def init_episode(
        self,
        episode_id,
    ):
        self.episode_dir = os.path.join(self.output_dir, episode_id)
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")
        eps_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        # 新增合并可视化目录
        eps_merge_dir = os.path.join(self.episode_dir, "merge_visualization")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)
        os.makedirs(eps_snapshot_dir, exist_ok=True)
        os.makedirs(eps_merge_dir, exist_ok=True)  # 创建合并可视化目录

        return self.episode_dir, eps_frontier_dir, eps_snapshot_dir

    def init_subtask(
        self,
        subtask_id,
        goal_type,
        subtask_goal,
        pts,
        scene: Scene,  # used to get image goal observation and use its pathfinder
        tsdf_planner: TSDFPlanner,
    ):
        # determine the navigation goals
        goal_category = subtask_goal[0]["object_category"]
        goal_obj_ids = [x["object_id"] for x in subtask_goal]
        goal_obj_ids = [int(x.split("_")[-1]) for x in goal_obj_ids]
        if goal_type != "object":
            assert len(goal_obj_ids) == 1, f"{len(goal_obj_ids)} != 1"

        goal_positions = [x["position"] for x in subtask_goal]
        goal_positions_voxel = [tsdf_planner.habitat2voxel(p) for p in goal_positions]

        viewpoints = [
            view_point["agent_state"]["position"]
            for goal in subtask_goal
            for view_point in goal["view_points"]
        ]
        # get the shortest distance from current position to the viewpoints
        all_distances = []
        for viewpoint in viewpoints:
            path = habitat_sim.ShortestPath()
            path.requested_start = pts
            path.requested_end = viewpoint
            found_path = scene.pathfinder.find_path(path)
            if not found_path:
                all_distances.append(np.inf)
            else:
                all_distances.append(path.geodesic_distance)
        gt_subtask_explore_dist = min(all_distances) + 1e-6

        self.subtask_object_observe_dir = os.path.join(
            self.output_dir, subtask_id, "object_observations"
        )
        if os.path.exists(self.subtask_object_observe_dir):
            os.system(f"rm -r {self.subtask_object_observe_dir}")
        os.makedirs(self.subtask_object_observe_dir, exist_ok=False)

        # Prepare metadata for the subtask
        subtask_metadata = {
            "question_id": subtask_id,
            "question": None,
            "image": None,
            "answer": goal_category,
            "goal_obj_ids": goal_obj_ids,  # this is a list of obj id, since for object class type, there will be multiple target objects
            "class": goal_category,
            "goal_positions_voxel": goal_positions_voxel,  # also a list of positions for possible multiple objects
            "task_type": goal_type,
            "viewpoints": viewpoints,
            "gt_subtask_explore_dist": gt_subtask_explore_dist,
        }
        # format question according to the goal type
        if goal_type == "object":
            subtask_metadata["question"] = f"Can you find the {goal_category}?"
        elif goal_type == "description":
            subtask_metadata["question"] = (
                f"Could you find the object exactly described as the '{subtask_goal[0]['lang_desc']}'?"
            )
        else:
            subtask_metadata["question"] = (
                f"Could you find the exact object captured at the center of the following image? You need to pay attention to the environment and find the exact object."
            )
            view_pos_dict = subtask_goal[0]["view_points"][0]["agent_state"]
            obs, _ = scene.get_observation(
                pts=view_pos_dict["position"], rotation=view_pos_dict["rotation"]
            )
            plt.imsave(
                os.path.join(self.output_dir, subtask_id, "image_goal.png"),
                obs["color_sensor"],
            )
            subtask_metadata["image"] = f"{self.output_dir}/{subtask_id}/image_goal.png"

        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack(
            [self.pts_voxels, tsdf_planner.habitat2voxel(pts)[:2]]
        )
        self.subtask_explore_dist = 0.0

        return subtask_metadata

    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.subtask_explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )

    def save_topdown_visualization(
        self, global_step, subtask_id, subtask_metadata, goal_obj_ids_mapping, fig
    ):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)

        ax1 = fig.axes[0]
        ax1.plot(
            self.pts_voxels[:-1, 1], self.pts_voxels[:-1, 0], linewidth=1, color="white"
        )
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)

        # add target object bbox
        for goal_id, goal_pos_voxel in zip(
            subtask_metadata["goal_obj_ids"], subtask_metadata["goal_positions_voxel"]
        ):
            color = "green" if len(goal_obj_ids_mapping[goal_id]) > 0 else "red"
            ax1.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)

        fig.tight_layout()
        # 返回保存路径，便于后续合并
        save_path = os.path.join(visualization_path, f"{global_step}_{subtask_id}.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def save_frontier_visualization(
        self,
        global_step,
        subtask_id,
        tsdf_planner: TSDFPlanner,
        max_point_choice: Union[SnapShot, Frontier],
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        episode_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        if type(max_point_choice) == SnapShot:
            num_images += 1
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")
                elif i == num_images - 1 and type(max_point_choice) == SnapShot:
                    img_path = os.path.join(
                        episode_snapshot_dir, max_point_choice.image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    axs[h_idx, w_idx].set_title("Snapshot Chosen")
        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        # 返回保存路径，便于后续合并
        save_path = os.path.join(frontier_video_path, f"{global_step}_{subtask_id}.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    # def save_merged_visualization(self, global_step, subtask_id, frontier_path, topdown_path):
    #     """
    #     将frontier可视化和topdown可视化合并为一张图
    #     左侧：frontier可视化结果
    #     右侧：topdown可视化结果
    #     """
    #     assert self.episode_dir is not None, "episode目录未初始化"
    #     # 构建visualization和frontier_video的目录路径
    #     visualization_path = os.path.join(self.episode_dir, "visualization")
    #     frontier_video_path = os.path.join(self.episode_dir, "frontier_video")  
    #         # 构建具体的文件路径
    #     visualization_file = f"{global_step}_{subtask_id}.png"
    #     frontier_video_file = f"{global_step}_{subtask_id}.png"  

    #     topdown_path = os.path.join(visualization_path, visualization_file)
    #     frontier_path = os.path.join(frontier_video_path, frontier_video_file)        
    #     # 合并结果保存目录（与frontier等目录同级）

    #     # 检查文件是否存在
    #     if not os.path.exists(topdown_path):
    #         print(f"Warning: Topdown visualization not found at {topdown_path}")
    #         return None
    #     if not os.path.exists(frontier_path):
    #         print(f"Warning: Frontier visualization not found at {frontier_path}")
    #         return None
    #     # 创建合并结果保存目录   
    #     merged_dir = os.path.join(self.episode_dir, "merge_visualization")
    #     os.makedirs(merged_dir, exist_ok=True)

    #     # 读取两张图片
    #     try:
    #         topdown_img = matplotlib.image.imread(topdown_path)
    #         frontier_img = matplotlib.image.imread(frontier_path)

    #     except Exception as e:
    #         logging.error(f"读取可视化图片失败: {e}")
    #         return None

    #     # 创建合并图像（1行2列布局）
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
    #     # 左侧显示frontier可视化
    #     ax1.imshow(frontier_img)
    #     ax1.set_title("Frontier Visualization", fontsize=14)
    #     ax1.axis("off")
        
    #     # 右侧显示topdown可视化
    #     ax2.imshow(topdown_img)
    #     ax2.set_title("Top-Down Visualization", fontsize=14)
    #     ax2.axis("off")
        
    #     # 添加全局标题
    #     fig.suptitle(f"Step: {global_step} | Subtask: {subtask_id}", fontsize=16)
        
    #     # 调整布局并保存
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为全局标题预留空间
    #     merged_save_path = os.path.join(merged_dir, f"{global_step}_{subtask_id}.png")
    #     plt.savefig(merged_save_path, bbox_inches="tight", dpi=300)
    #     plt.close()
        
    #     return merged_save_path
    def save_merged_visualization(self, global_step, subtask_id, frontier_path, topdown_path):
        """
        合并frontier和topdown的可视化结果并保存
        Args:
            global_step: 当前全局步数
            subtask_id: 当前子任务ID
            frontier_path: frontier可视化图像路径
            topdown_path: topdown可视化图像路径
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        
        assert self.episode_dir is not None, "请先初始化episode"
        assert os.path.exists(frontier_path), f"frontier图像路径不存在: {frontier_path}"
        assert os.path.exists(topdown_path), f"topdown图像路径不存在: {topdown_path}"
        
        # 获取合并可视化保存目录
        merge_dir = os.path.join(self.episode_dir, "merge_visualization")
        os.makedirs(merge_dir, exist_ok=True)
        
        # 读取图像
        try:
            frontier_img = Image.open(frontier_path).convert("RGB")
            topdown_img = Image.open(topdown_path).convert("RGB")
        except Exception as e:
            logging.error(f"读取可视化图像失败: {e}")
            return
        
        # 调整图像大小以适应合并（保持宽高比）
        max_height = max(frontier_img.height, topdown_img.height)
        frontier_img = frontier_img.resize((int(frontier_img.width * max_height / frontier_img.height), max_height))
        topdown_img = topdown_img.resize((int(topdown_img.width * max_height / topdown_img.height), max_height))
        
        # 创建合并图像（横向拼接）
        merged_width = frontier_img.width + topdown_img.width
        merged_height = max_height
        merged_img = Image.new('RGB', (merged_width, merged_height))
        merged_img.paste(frontier_img, (0, 0))
        merged_img.paste(topdown_img, (frontier_img.width, 0))
        
        # 保存合并图像
        merged_path = os.path.join(merge_dir, f"merged_{global_step}_{subtask_id}.png")
        merged_img.save(merged_path)
        logging.info(f"合并可视化结果已保存至: {merged_path}")
        return merged_path

    # 新增：处理可视化并合并的方法（用于统一调用）
    def process_and_merge_visualizations(
        self, 
        global_step, 
        subtask_id, 
        subtask_metadata, 
        goal_obj_ids_mapping, 
        topdown_fig,
        tsdf_planner,
        max_point_choice,
        global_caption
    ):
        """
        处理frontier和topdown可视化并合并结果
        """
        # 保存topdown可视化并获取路径
        topdown_path = self.save_topdown_visualization(
            global_step=global_step,
            subtask_id=subtask_id,
            subtask_metadata=subtask_metadata,
            goal_obj_ids_mapping=goal_obj_ids_mapping,
            fig=topdown_fig
        )
        
        # 保存frontier可视化并获取路径
        frontier_path = self.save_frontier_visualization(
            global_step=global_step,
            subtask_id=subtask_id,
            tsdf_planner=tsdf_planner,
            max_point_choice=max_point_choice,
            global_caption=global_caption
        )
        
        # 合并可视化结果
        if topdown_path and frontier_path:
            self.save_merged_visualization(
                global_step=global_step,
                subtask_id=subtask_id,
                frontier_path=frontier_path,
                topdown_path=topdown_path
            )
