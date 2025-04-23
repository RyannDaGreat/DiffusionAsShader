#Run this script to allow you to use more frequent checkpoint saving without running out of ram!

from rp import *

while True:
    set_current_directory(
        "/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/cogshader_inv-avatar-physics_steps_2000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4"
    )
    ans = os.listdir()
    ans = sorted_by_number(ans)
    keeping = [ans[0], ans[-1]] + [x for x in ans if ends_with_any(x, "00", "50")]
    deleting = sorted_by_number(set(ans) - set(keeping))
    for x in deleting:
        fansi_print(f"Deleting {x}", "red orange bold on black black italic")
        rp.r._run_sys_command("rm", "-rf", x)
