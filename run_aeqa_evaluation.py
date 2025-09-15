"""
主程序入口，用于处理视觉语言导航任务。
该程序加载模型和数据，处理每个问题，并执行导航步骤。
这段代码是 AEQA (Active Embodied Question Answering) 任务的主程序入口 (main 函数)。
它的核心功能是加载模型、读取问题数据集，然后对每个问题执行一个主动探索和视觉问答的过程。
核心流程概述：
1、初始化: 加载配置、设置日志、加载视觉模型 (YOLO, SAM, CLIP)。
2、数据加载: 读取问题列表 (包含场景、问题、答案、目标位置等)。
3、问题处理: 对每个问题，执行以下步骤：
单个问题处理:

   加载对应的 3D 场景 (Habitat-Sim)。
    初始化 TSDF 地图 (用于构建占据栅格地图和前沿探索)。
    初始化 Scene 对象 (用于管理场景图、对象检测、记忆快照)。
    主循环 (导航步):
        (1) 观察: 在当前位置和多个角度获取观测 (RGB, Depth)。
        (2) 更新场景: 使用观测更新场景图 (检测物体、分割、CLIP 特征)。
        (3) 更新地图: 将观测融合到 TSDF 地图中，更新前沿区域。
        (4) 决策: 如果需要，查询 VLM (如 GPT) 决定下一步导航目标 (前沿或物体快照)。
        (5) 移动: 执行一步导航移动。
        (6) 检查结束: 判断是否到达目标物体，如果到达则任务成功，结束循环。
 

结果记录: 记录每个问题的处理结果 (成功/失败、探索距离、VLM 回答等)。
最终汇总: 汇总所有问题的结果。



"""
import os  # 导入操作系统接口模块

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf# 用于加载和解析 YAML 配置文件
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt

# 导入项目自定义模块和第三方库
import open_clip # Open-source CLIP 模型库
from ultralytics import SAM, YOLOWorld # Ultralytics 提供的 SAM 和 YOLO-World 模型

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_aeqa import Scene
from src.utils import resize_image, get_pts_angle_aeqa
from src.query_vlm_aeqa import query_vlm_for_response
from src.logger_aeqa import Logger
from src.const import *


def main(cfg, start_ratio=0.0, end_ratio=1.0):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)


# 获取图像尺寸和相机内参
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width) # 获取场景边界以及计算相机内参矩阵

 # 设置随机种子以保证可复现性
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 2. Load dataset加载数据集
    questions_list = json.load(open(cfg.questions_list_path, "r")) # 从 JSON 文件加载问题列表
    total_questions = len(questions_list)
    # sort the data according to the question id 根据 question_id 对问题进行排序
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")

    # only process a subset of the questions只处理指定比例范围内的问题 (用于并行处理或测试)
    questions_list = questions_list[
        int(start_ratio * total_questions) : int(end_ratio * total_questions)
    ]
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")

    # 3. load detection and segmentation models加载检测和分割模型
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    # 4. load CLIP model加载 CLIP 模型
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", 
        pretrained="/home/ubuntu/projects/3D-Mem/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger初始化日志   
    logger = Logger(
        cfg.output_dir, # 输出目录
        start_ratio,    # 开始比例
        end_ratio,     # 结束比例
        len(questions_list),  # 问题总数
        voxel_size=cfg.tsdf_grid_size,  # TSDF 体素大小，用于记录探索距离
    )

    # Run all questions 遍历处理每个问题
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]
        # 如果问题已经处理过 (成功或失败)，则跳过
        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue

        # 如果场景 ID 无效，则跳过
        if any([invalid_scene_id in scene_id for invalid_scene_id in INVALID_SCENE_ID]):
            logging.info(f"Skip invalid scene {scene_id}")
            continue
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        # 提取问题、答案和目标位置/角度
        question = question_data["question"]
        answer = question_data["answer"]
        pts, angle = get_pts_angle_aeqa(
            question_data["position"], question_data["rotation"]
        )

        # 6、load scene加载场景
        try:
            del scene # 尝试删除之前的场景对象
        except:
            pass
        # 创建新的 Scene 对象，加载场景并初始化模型
        scene = Scene(
            scene_id,
            cfg,
            cfg_cg,
            detection_model,
            sam_predictor,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
        )

        # 7、initialize the TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=get_scene_bnds(scene.pathfinder, floor_height=pts[1])[0], # 获取场景边界
            voxel_size=cfg.tsdf_grid_size,  # 体素大小
            floor_height=pts[1],    # 地面高度
            floor_height_offset=0,  # 地面高度偏移
            pts_init=pts,                # 初始位置
            init_clearance=cfg.init_clearance * 2,       # 初始清除距离
            save_visualization=cfg.save_visualization, # 是否保存可视化结果
        )
      
        # 8、initialize the logger
        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],     # 初始化位置
            )
        )

        logging.info(f"\n\nQuestion id {question_id} initialization successful!")

        # 8. 开始导航主循环 run steps
        task_success = False        # 任务是否成功
        cnt_step = -1                # 步数计数器

        gpt_answer = None        # GPT 答案
        n_filtered_snapshots = 0     # 被过滤的快照数量                       
        while cnt_step < cfg.num_step - 1:  # 最多执行 cfg.num_step 步
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

            # (1) Observe the surroundings, update the scene graph and occupancy map
            # Determine the viewing angles for the current step
            # (1) 观察周围环境，更新场景图和占据地图
            # 确定当前步的观测角度
            if cnt_step == 0:
                # 第 0 步，使用 Phase 2 的参数 (更细致的观察)
                angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_2
            else:
                # 其他步，使用 Phase 1 的参数 (更侧重探索)
                angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_1
                # 生成所有观测角度
            all_angles = [
                angle + angle_increment * (i - total_views // 2)
                for i in range(total_views)
            ]
            # 将主视角放在最后，避免覆盖问题
            # Let the main viewing angle be the last one to avoid potential overwriting problems
            main_angle = all_angles.pop(total_views // 2)
            all_angles.append(main_angle)

            rgb_egocentric_views = []
            all_added_obj_ids = (
                []
            )  # Record all the objects that are newly added in this step
            for view_idx, ang in enumerate(all_angles):
                # For each view
                obs, cam_pose = scene.get_observation(pts, ang) # 获取观测数据和相机位姿
                rgb = obs["color_sensor"] # RGB 图像
                depth = obs["depth_sensor"]  # 深度图

                obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                with torch.no_grad(): # 禁用梯度计算，节省资源
                    # Concept graph pipeline update 更新场景图：检测、分割、特征提取、添加到场景图
                    annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
                        image_rgb=rgb[..., :3],# 去掉 Alpha 通道
                        depth=depth,
                        intrinsics=cam_intr, # 相机内参
                        cam_pos=cam_pose, # 相机位姿
                        pts=pts, # 代理位置
                        pts_voxel=tsdf_planner.habitat2voxel(pts),# 代理位置的体素坐标
                        img_path=obs_file_name,
                        frame_idx=cnt_step * total_views + view_idx,# 帧索引
                        target_obj_mask=None,# AEQA 中不使用特定目标掩码
                    )
                    # 调整图像尺寸用于 VLM 提示
                    resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                    scene.all_observations[obs_file_name] = resized_rgb
                    # 添加到 VLM 视图列表
                    rgb_egocentric_views.append(resized_rgb)
                    
                     # 保存观测图像
                    if cfg.save_visualization:
                        plt.imsave(
                            os.path.join(eps_snapshot_dir, obs_file_name), annotated_rgb
                        )
                    else:
                        plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                    all_added_obj_ids += added_obj_ids # 记录新添加的对象 ID

                # Clean up or merge redundant objects periodically 定期清理或合并场景中的冗余对象
                scene.periodic_cleanup_objects(
                    frame_idx=cnt_step * total_views + view_idx, pts=pts
                )

                # Update depth map, occupancy map 定期清理或合并场景中的冗余对象
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=pose_habitat_to_tsdf(cam_pose), # 转换位姿格式
                    obs_weight=1.0, # 观测权重
                    margin_h=int(cfg.margin_h_ratio * img_height), # 图像边缘裁剪
                    margin_w=int(cfg.margin_w_ratio * img_width), 
                    explored_depth=cfg.explored_depth,# 已探索深度阈值
                )

            # (2) Update Memory Snapshots with hierarchical clustering使用层次聚类更新记忆快照 (Snapshots)
            # Choose all the newly added objects as well as the objects nearby as the cluster targets 选择新添加的对象以及附近的对象作为聚类目标
            #目的: 这行代码是一个安全检查。虽然可能性不大，但为了确保万无一失，它会重新检查这些新添加的 obj_id 是否确实存在于 scene.objects 字典中。
            # 如果因为某些原因（比如在 update_scene_graph 和现在之间有清理操作）某个ID被移除了，它就会被过滤掉。这确保了后续处理的对象都是有效的。
            
            all_added_obj_ids = [
                obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects
            ]
            # 查找并添加附近的物体ID，遍历当前场景中所有的物体
            for obj_id, obj in scene.objects.items():
                # 计算物体中心与代理当前位置在水平面上（XZ平面）的距离
                if (
                    np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]]) # 计算水平距离
                    < cfg.scene_graph.obj_include_dist + 0.5 # 判断是否在扩展范围内
                ):
                    all_added_obj_ids.append(obj_id) # 如果距离小于阈值，则将该物体的ID也加入待聚类列表
                    # 3. 执行实际的快照更新操作
            scene.update_snapshots(
                # 传入去重后的物体ID集合以及生成的最小检测次数阈值
                obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
            )
            logging.info(
                f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
            )

            # (3) Update the Frontier Snapshots更新快照
            update_success = tsdf_planner.update_frontier_map(
                pts=pts,
                cfg=cfg.planner,
                scene=scene,
                cnt_step=cnt_step,
                save_frontier_image=cfg.save_visualization,
                eps_frontier_dir=eps_frontier_dir,
                prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
            )
            if not update_success:
                logging.info("Warning! Update frontier map failed!")
                if cnt_step == 0:  # if the first step fails, we should stop
                    logging.info(
                        f"Question id {question_id} invalid: update_frontier_map failed!"
                    )
                    break

            # (4) Choose the next navigation point by querying the VLM# 通过查询 VLM 选择下一个导航点
            if cfg.choose_every_step:
                # if we choose to query vlm every step, we clear the target point every step如果配置为每步都查询 VLM，则清除之前的目标点
                if (
                    tsdf_planner.max_point is not None
                    and type(tsdf_planner.max_point) == Frontier
                ):
                    # reset target point to allow the model to choose again
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None
            # 如果没有目标点，则查询 VLM
            if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                # query the VLM for the next navigation point, and the reason for the choice查询 VLM 获取导航决策和回答
                vlm_response = query_vlm_for_response(
                    question=question,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    rgb_egocentric_views=rgb_egocentric_views,# 提供给 VLM 的图像
                    cfg=cfg,
                    verbose=True,
                )
                if vlm_response is None:
                    logging.info(
                        f"Question id {question_id} invalid: query_vlm_for_response failed!"
                    )
                    break
                # 解析 VLM 响应
                max_point_choice, gpt_answer, n_filtered_snapshots = vlm_response

                # set the vlm choice as the navigation target将 VLM 的选择设置为导航目标
                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts,
                    objects=scene.objects,
                    cfg=cfg.planner,
                    pathfinder=scene.pathfinder,
                    random_position=False,
                )
                if not update_success:
                    logging.info(
                        f"Question id {question_id} invalid: set_next_navigation_point failed!"
                    )
                    break

            # (5) Agent navigate to the target point for one step
            return_values = tsdf_planner.agent_step(
                pts=pts,
                angle=angle,
                objects=scene.objects,
                snapshots=scene.snapshots,
                pathfinder=scene.pathfinder,
                cfg=cfg.planner,
                path_points=None,
                save_visualization=cfg.save_visualization,
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break

            # update agent's position and rotation更新代理的位置和角度
            pts, angle, pts_voxel, fig, _, target_arrived = return_values
            logger.log_step(pts_voxel=pts_voxel)  #记录步数和位置
            logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")# 打印当前位置和探索距离

            # sanity check about objects, scene graph, snapshots, ...对场景状态进行健全性检查
            scene.sanity_check(cfg=cfg)

            if cfg.save_visualization:
                # save the top-down visualization
                logger.save_topdown_visualization(
                    cnt_step=cnt_step,
                    fig=fig,
                )
                # save the visualization of vlm's choice at each step
                logger.save_frontier_visualization(
                    cnt_step=cnt_step,
                    tsdf_planner=tsdf_planner,
                    max_point_choice=max_point_choice,
                    global_caption=f"{question}\n{answer}",
                )

            # (6) Check if the agent has arrived at the target to finish the question
            if type(max_point_choice) == SnapShot and target_arrived:
                # when the target is a snapshot, and the agent arrives at the target
                # we consider the question is finished and save the chosen target snapshot
                snapshot_filename = max_point_choice.image.split(".")[0]
                os.system(
                    f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                )

                task_success = True
                logging.info(
                    f"Question id {question_id} finished after arriving at target!"
                )
                break

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=logger.explore_dist, # 总探索距离
            gpt_answer=gpt_answer, # VLM 回答
            n_filtered_snapshots=n_filtered_snapshots, # 被过滤快照数
            n_total_snapshots=len(scene.snapshots), # 总快照数
            n_total_frames=len(scene.frames), # 总帧数
        )
        
         # 打印场景图和问答信息
        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Prediction: {gpt_answer}")
        scene.print_scene_graph() # 打印构建的场景图

        # update the saved results after each episode
        logger.save_results()

        if not cfg.save_visualization:
            # clear up the stored images to save memory
            os.system(f"rm -r {episode_dir}")
     # 10. 最终保存和汇总结果
    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path# 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging 加载主配置文件
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive 创建目录
    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log"
    )
    # 复制配置文件到输出目录
    os.system(f"cp {args.cfg_file} {cfg.output_dir}")
    
    # 自定义日志格式，显示运行时间
    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    # 配置日志记录到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, args.start_ratio, args.end_ratio)
