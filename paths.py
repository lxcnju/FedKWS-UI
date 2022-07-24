import os

try:
    from naie.context import Context
    print(Context.get_project_path())
    project_path = Context.get_project_path()
    data_dir = os.path.join(project_path, "Dataset", "speech")
    cur_dir = os.path.join(project_path, "Algorithm", "algo-speech")

    speech_commands_fdir = os.path.join(data_dir, "speechcommands", "data")

except Exception:
    data_dir = r"C:\Workspace\work\datasets"
    cur_dir = "./"
    speech_commands_fdir = os.path.join(data_dir, "SpeechCommands")

    if not os.path.exists(data_dir):
        data_dir = r"/home/yjwang/datasets/"
        pretrain_dir = r"/home/yjwang/pretrain/"
        speech_commands_fdir = os.path.join(
            data_dir, "speech"
        )

    if not os.path.exists(data_dir):
        data_dir = r"/home/lixc/datasets/"
        pretrain_dir = r"/home/lixc/pretrain/"
        speech_commands_fdir = os.path.join(
            data_dir, "SpeechCommands"
        )

save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt_dir = os.path.join(cur_dir, "ckpts")
