# about habitat scene
INVALID_SCENE_ID = []

# about chatgpt api
# END_POINT = "https://ai.nengyongai.cn/v1"
# OPENAI_KEY = "sk-i68HfKeiendMDHobBHBWyPfl0Sd4fIKoORqPMT82IcP9tlpe"
import os
# # # # Qwen-VL模型配置参数
# # # # # 从环境变量中获取基础URL，如果未设置则使用默认值localhost
# QWEN_VL_BASE_URL = os.getenv("QWEN_VL_BASE_URL", "http://localhost:7080/v1")
# # # # # # 从环境变量中获取API密钥，如果未设置则使用默认值dummy
# QWEN_VL_API_KEY  = os.getenv("QWEN_VL_API_KEY",  "dummy")
# # # # # # 从环境变量中获取模型路径，如果未设置则使用指定路径下的Qwen2.5-VL-7B-Instruct模型
# QWEN_VL_MODEL    = os.getenv("QWEN_VL_MODEL",)
# # # #     "/home/ubuntu/projects2/Qwen2.5-VL/Qwen/Qwen2.5-VL-7B-Instruct")

# # InternVL2_5-8B 模型配置参数
# # 从环境变量中获取基础URL，如果未设置则使用默认值 0.0.0.0:8000
INTERNVL_BASE_URL = os.getenv("INTERNVL_BASE_URL", "http://0.0.0.0:8000/v1")
# 从环境变量中获取API密钥，如果未设置则使用默认值 dummy (vLLM通常不需要或接受任意值)
INTERNVL_API_KEY  = os.getenv("INTERNVL_API_KEY",  "dummy")
# # # 从环境变量中获取模型名称，如果未设置则使用默认的模型标识符 
INTERNVL_MODEL    = os.getenv("INTERNVL_MODEL", "OpenGVLab/InternVL2_5-8B")