from models.SMPmodel import load_model_from_checkpoint
from Script.Train import get_best_checkpoint
from Script.Real import test_camera_low_res, batch_test_images

best_paths = get_best_checkpoint("checkpoints", "camvid_segmentation")
model = load_model_from_checkpoint(best_paths)

# result = batch_test_images(
#     model=model,
#     image_dir="data/CamVid/test",
#     output_dir="ResOutput"
# )
# print(result)

test_camera_low_res(
    model=model,
    camera_id=0  # 默认摄像头
)
