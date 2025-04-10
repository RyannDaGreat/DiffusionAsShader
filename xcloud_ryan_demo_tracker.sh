#Meant to run on XCloud

export HF_HUB_OFFLINE=1 

rclone sync --progress /home/jupyter/CleanCode/CloudSync/home_cache ~/.cache
bash ~/CleanCode/Management/sync_projects.bash
bash ryan_demo_tracker.sh

bash ./generate_tracking_only.sh
# bash ./ryan_demo_tracker.sh
