from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging

def get_log_file(dir):
    for f in os.listdir(dir):
        file_path = os.path.join(dir, f)
        if os.path.isfile(file_path) and file_path.endswith("train.log"):
            print(file_path)
            return file_path

def get_all_subdirs(dir):
    x = []
    for f in os.listdir(dir):
        dir_path = os.path.join(dir, f)
        if os.path.isdir(dir_path):
            x.append(dir_path)
    return x

def plot_fig(acc, tags, exp_dir, ylabel):
    plt.axis("on")
    plt.xlabel("epochs")
    plt.ylabel(ylabel)
    for tag in tags:
        df = pd.DataFrame(acc.Scalars(tag))
        plt.plot( df["step"], df["value"])
    plt.legend(tags)
    plt.savefig(f"{ os.path.join(exp_dir, '_'.join(tags)) }.png", bbox_inches='tight', pad_inches=0.1) 
    plt.close(fig='all')


if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', filename='experiments.log', level = logging.INFO, filemode = 'w')
    # base_dir = os.path.join("log","babypose","multi_out_pose_hrnet", "lr_75_step_30_50_fact_0.33")
    base_dir = "/home/alecacciatore/HRNet-Human-Pose-Estimation_ale/output/babypose/multi_out_pose_hrnet/"
    dir_list = []
    exps = get_all_subdirs(base_dir)

    for exp_dir in exps:
        logging.info(f"{'-'*30}\nexperiment: {exp_dir}\n{'-'*30}")
        try:
            scalars_path = get_log_file(exp_dir)
            main_acc = EventAccumulator(scalars_path)
            main_acc.Reload()
            
            tmp_df = pd.DataFrame(main_acc.Scalars("valid_loss"))
            logging.info(f"final epoch: {tmp_df.shape[0] - 1}")

            if "train_loss" in main_acc.Tags():
                plot_fig(main_acc, ["train_loss", "valid_loss"], exp_dir, "loss")
                plot_fig(main_acc, ["train_loss_teacher", "train_loss_hard", "train_loss_soft"], exp_dir, "train_losses")
                plot_fig(main_acc, ["valid_loss_teacher", "valid_loss_hard", "valid_loss_soft"], exp_dir, "valid_losses")
            
            metrics = get_all_subdirs(exp_dir)
            
            for f in metrics:
                tag = os.path.basename(os.path.normpath(f))
                df_acc = pd.DataFrame(main_acc.Scalars(tag+"_acc"))

                logging.info(f"\n{tag}")
                ap_dir = os.path.join(f, "AP")
                ar_dir = os.path.join(f, "AR")
                
                path = get_log_file(ap_dir)
                acc = EventAccumulator(path)
                acc.Reload()
                plot_fig(acc, [tag], ap_dir, "AP")
                df_AP = pd.DataFrame(acc.Scalars(tag))
                max_AP = df_AP['value'].max()
                max_index = df_AP['value'].idxmax()
                max_AP_epoch = df_AP['step'][max_index]
                logging.info(f"max AP {max_AP:.4f} at epoch {max_AP_epoch}")
                
                path = get_log_file(ar_dir)
                acc = EventAccumulator(path)
                acc.Reload()
                df_AR = pd.DataFrame(acc.Scalars(tag))
                max_AR = df_AR['value'][max_index]
                logging.info(f"AR {max_AR:.4f}")

                max_acc = df_acc['value'][max_index]
                logging.info(f"PCK@0.5 {max_acc:.4f}")

        except Exception as e:
            print(e)