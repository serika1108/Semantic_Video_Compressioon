python test_video.py --i_frame_model_name ConvChARM \
--i_frame_model_path  checkpoints/ConvChARM.pth --test_config dataset_config_others.json \
--cuda True --output_json_result_path  DCVC_result_psnr.json \
--model_path checkpoints/model_dcvc_quality_0_psnr.pth \
--write_stream False --write_recon_frame False

python test_video.py --i_frame_model_name ConvChARM \
--i_frame_model_path  checkpoints/ConvChARM.pth --test_config dataset_config_HEVC.json \
--cuda True --output_json_result_path  DCVC_result_psnr.json \
--model_path checkpoints/model_dcvc_quality_0_psnr.pth \
--write_stream False --write_recon_frame False